from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from evals.agents.memo_structure import MemoStructure
from evals.agents.base import init_global_tracker
from evals.utils.hire_agent import Agent
from core.memo_manager import Memo_Manager
from core.meta_agent_prompt import build_analysis_prompt, build_generate_new_code_prompt, build_reflection_prompt
import os
import asyncio
import json
from evals.logger import get_logger
from datetime import datetime
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import importlib

log = get_logger("main")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "evals" / "logs"
RECORDER = {
    'alfworld': "ALFWorldRecorder",
    'minihack': "MiniHackRecorder",
    'textworld': "TextworldRecorder",
    'babaisai': "BabaisaiRecorder"
}

class MetaAgent:
    def __init__(self, task_type, meta_model = 'gpt-4.1', execution_model = 'gpt-4o-mini', status = 'train', history_ckpt_path: Optional[str] = None):
        self.memo_manager = Memo_Manager(
            task_type = task_type,
            status = status,
            history_ckpt_path = history_ckpt_path
        )
        self.history_ckpt_path = history_ckpt_path
        self.task_type = task_type
        self.examine_trial = 3
        self.meta_model = meta_model
        self.execution_model = execution_model

    def read_memo_info(self, memo_SHA: str, mode = 'test'):

        assert memo_SHA in self.memo_manager.memo_db

        source_code: str = self.memo_manager.read_source_code(memo_SHA)
        evals = self.memo_manager.read_eval_result(memo_SHA, mode)
        evals["source_code"] = source_code
        parent = self.memo_manager.memo_db[memo_SHA]['parent']
        if parent:
            try:
                evals['improve_example'] = {
                    'source_code': self.memo_manager.read_source_code(parent),
                    'suggestion': self.memo_manager.memo_db[memo_SHA]['suggestion'],
                    'improve_score': self.memo_manager.memo_db[memo_SHA]['improve_score']
                }
            except Exception as e:
                log.warning(f"Fail to form imporve sample: {e}")
        return evals

    async def analyze_memo_structure(self, memo_SHA: str):
        """
        Analyze current MemoStructure.
        Returns a string summary including:
        - Registered layers
        - Pipeline steps
        - Reward and performance hints
        """
        # Ask the agent to analyze and suggest improvements
        memo_info = self.read_memo_info(memo_SHA, mode = 'eval')
        sys_msg, user_msg, output_schema = build_analysis_prompt(memo_info, self.task_type) 
        analysis_agent = Agent(
            system_prompt = sys_msg,
            output_schema = output_schema,
            model = self.meta_model
        )
        result = await analysis_agent.ask(user_msg, reasoning_effort='medium')
        # log.info(json.dumps(result))
        return result, memo_info

    async def generate_new_code(self, analysis_result: Dict[str, Any] = {}, memo_info: Dict[str, Any] = {}) -> str:
        """
        Generate a new SubMemoLayer code based on instruction.
        Returns code_str.
        """
        module = importlib.import_module(f"envs_archive.{self.task_type}_envs")  
        recorder = getattr(module, RECORDER[self.task_type])
        sys_msg, user_msg = build_generate_new_code_prompt(memo_info = memo_info, 
                                                            analysis_result = analysis_result, 
                                                            recorder = recorder,
                                                            task_type = self.task_type)

        gen_code_agent = Agent(system_prompt = sys_msg, model = self.meta_model)
        code_str = await gen_code_agent.ask(user_msg, reasoning_effort='medium')
        return code_str
    
    async def examine_new_code(self, 
                            code_str: str, 
                            eval_type: str, 
                            train_size = 30, 
                            batch_max_update_concurrent = 10, 
                            batch_max_retrieve_concurrent = 10,
                            update_size: Optional[int] = None, 
                            update_task: Optional[str] = None
                            ):

        memo_SHA = None
        module = importlib.import_module(f"envs_archive.{self.task_type}_envs")  
        recorder = getattr(module, RECORDER[self.task_type])
        for _ in range(self.examine_trial):
            log.info(f"Start examination for round {_}")
            try:
                success, examine_log, memo_SHA, code_str = await self.memo_manager.execute_memo_structure(
                    code_str = code_str, 
                    target_sha = memo_SHA, 
                    mode = 'test', 
                    eval_type = eval_type, 
                    model = self.execution_model,
                    train_size = train_size,
                    batch_max_update_concurrent = batch_max_update_concurrent,
                    batch_max_retrieve_concurrent = batch_max_retrieve_concurrent,
                    update_size = update_size,
                    update_task = update_task
                    )
            except Exception as e:
                success = False
                examine_log = e
            if success:
                break
            log.warning(f"Fail examination for round {_}")
            sys_msg, user_msg = build_reflection_prompt(code_str = code_str,
                                                            recorder = recorder,
                                                            error_msg = examine_log)
            reflect_fix_agent = Agent(system_prompt = sys_msg, model = self.meta_model)
            code_str = await reflect_fix_agent.ask(user_msg)
            log.info(f"retried finished for round {_}")
        if not success:
            raise RuntimeError(f"Fail to revise code in {self.examine_trial} attempt.")
        else:
            self.memo_manager.save_memo_structure(code_str, memo_SHA)
            return memo_SHA, code_str
            
    async def run_single_memo(self, 
                            memo_SHA, 
                            eval_type = 'sequential', 
                            train_size = 30, 
                            status = 'train', 
                            batch_max_update_concurrent = 10, 
                            batch_max_retrieve_concurrent = 10,
                            update_size: Optional[int] = None, 
                            update_task: Optional[str] = None, 
                            **kargs
                            ):
        
        if status == 'train':
            #updates visiting time for original memo
            self.memo_manager.update_visit_time(memo_SHA)

            # development of new structure
            log.info("[blue]━━━━━━━━━━━━━━━ STRUCTURE REFLECTION ━━━━━━━━━━━━━━━[/blue]")
            analysis_result, memo_info = await self.analyze_memo_structure(memo_SHA = memo_SHA)
            log.info(f"[blue][FINISH STRUCTURE REFLECTION]: Reflect example: {analysis_result.get('suggested_changes','')[0]['what'][:10]}...[/blue]")
            
            log.info("[blue]━━━━━━━━━━━━━━━ CODE GENERATION ━━━━━━━━━━━━━━━[/blue]")
            new_code = await self.generate_new_code(analysis_result = analysis_result, memo_info = memo_info)
            
            new_memo_SHA, new_code = await self.examine_new_code(
                                                            code_str = new_code, 
                                                            eval_type = eval_type, 
                                                            train_size = train_size, 
                                                            batch_max_update_concurrent = batch_max_update_concurrent, 
                                                            batch_max_retrieve_concurrent = batch_max_update_concurrent
                                                            )

            self.memo_manager.update_analysis(memo_sha=new_memo_SHA, suggestion=analysis_result)
            log.info(f"[blue][FINISH CODE GENERATION] Code SHA: {new_memo_SHA} | Code example: {new_code[:30]}...[/blue]")
        else:
            # eval the memo
            new_memo_SHA = memo_SHA

        log.info("[blue]━━━━━━━━━━━━━━━ CODE EXECUTION ━━━━━━━━━━━━━━━[/blue]")
        # run evaluation of new memo structure and updates
        _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
            target_sha = new_memo_SHA, 
            mode = 'eval', 
            eval_type = eval_type, 
            model = self.execution_model,
            train_size = train_size,
            status = status,
            batch_max_update_concurrent = batch_max_update_concurrent, 
            batch_max_retrieve_concurrent = batch_max_retrieve_concurrent,
            update_size = update_size,
            update_task = update_task
            )
        
        if status == 'train':
            self.memo_manager.update_reward(new_memo_SHA, eval_result.get('benchmark_eval_score',{}).get('benchmark_overall_eval_score',0.0))
            self.memo_manager.update_parent(new_memo_SHA, memo_SHA)
        log.info(f"[blue][FINISH CODE EXECUTION] Code SHA: {new_memo_SHA} | Reward: {eval_result.get('benchmark_eval_score',{}).get('benchmark_overall_eval_score',0.0)}[/blue]")


    async def forward(self, 
                    rollout_type: str = 'sequential', 
                    steps: int = 10, 
                    max_concurrent = 5, 
                    result_dir = "check", 
                    train_size = 30, 
                    batch_max_update_concurrent = 10, 
                    batch_max_retrieve_concurrent = 10,
                    update_size: Optional[int] = None, 
                    update_task: Optional[str] = None
                    ):
        if self.history_ckpt_path is None: # start from no ckpt
            _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
                        target_sha = 'no_mem', 
                        mode = 'eval', 
                        eval_type = rollout_type, 
                        model = self.execution_model,
                        train_size = train_size,
                        status = 'train',
                        batch_max_update_concurrent = 10, 
                        batch_max_retrieve_concurrent = 10,
                        update_size = 0,
                        update_task = update_task
                        )
            self.memo_manager.no_memo_reward = eval_result.get('benchmark_eval_score').get('benchmark_overall_eval_score') #baseline score
            # log.info(self.memo_manager.no_memo_reward)
            tracker = init_global_tracker()
            progress = Progress(
                        TextColumn(f"[bold yellow]Initial Memory Structure Construction"),
                        BarColumn(bar_width=None, complete_style="green"),
                        TextColumn("{task.completed}/{task.total}"),
                        TimeElapsedColumn(),
                        expand=True,
                    )
            task_id = progress.add_task("Running", total=2) 
            log.info("[blue]━━━━━━━━━━━━━━━ CODE GENERATION ━━━━━━━━━━━━━━━[/blue]")
            new_code = await self.generate_new_code()
            
            new_memo_SHA, new_code = await self.examine_new_code(
                                                code_str = new_code, 
                                                eval_type = rollout_type, 
                                                batch_max_update_concurrent = batch_max_update_concurrent,
                                                batch_max_retrieve_concurrent = batch_max_retrieve_concurrent,
                                                update_size = update_size,
                                                train_size = train_size
                                                )

            log.info(f"[blue][FINISH CODE GENERATION] Code SHA: {new_memo_SHA} | Code example: {new_code[:30]}...[/blue]")

            log.info("[blue]━━━━━━━━━━━━━━━ CODE EXECUTION ━━━━━━━━━━━━━━━[/blue]")
            # run evaluation of new memo structure and updates
            _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
                target_sha = new_memo_SHA, 
                mode = 'eval', 
                eval_type = rollout_type,
                model = self.execution_model,
                train_size = train_size,
                batch_max_update_concurrent = batch_max_update_concurrent,
                batch_max_retrieve_concurrent = batch_max_retrieve_concurrent,
                update_size = update_size
                )
            self.memo_manager.update_reward(new_memo_SHA, eval_result.get('benchmark_eval_score',{}).get('benchmark_overall_eval_score',0.0))
            self.memo_manager.update_parent(new_memo_SHA, '')
            file_name = f"{result_dir}_{self.task_type}_{rollout_type}_{steps}_{timestamp}.json" #init new log path
            log.info(f"[blue][FINISH CODE EXECUTION] Code SHA: {new_memo_SHA} | Reward: {eval_result.get('benchmark_eval_score',{}).get('benchmark_overall_eval_score',0.0)}[/blue]")
        else:
            file_name = self.history_ckpt_path
        # run main loop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).resolve().parent.parent
        
        semaphore = asyncio.Semaphore(max_concurrent)
        for step in range(steps):
            with open(project_root/"logs"/file_name, "w", encoding="utf-8") as f:
                json.dump(self.memo_manager.memo_db, f, ensure_ascii=False, indent=2)
            result_path = project_root / "logs" / file_name
            log.info(f"[LOG RESULT] Results saved to: {result_path}")
            log.info(f"[blue]\n━━━━━━━━━━━━━━━ STEP [{step+1}] ━━━━━━━━━━━━━━━[/blue]")
            memo_SHA_list = self.memo_manager.select_structure()
            log.info(f"Select memos {memo_SHA_list}")

            async def sem_task(memo_SHA):
                async with semaphore:
                    try:
                        await self.run_single_memo(
                            memo_SHA, 
                            eval_type=rollout_type, 
                            batch_max_update_concurrent = batch_max_update_concurrent, 
                            batch_max_retrieve_concurrent = batch_max_retrieve_concurrent,
                            update_size = update_size, 
                            update_task = update_task,
                            train_size = train_size
                            )
                        progress.update(task_id, advance=1)
                    except Exception as e:
                        log.warning(f"Memo {memo_SHA} failed: {e}")
            await asyncio.gather(*(sem_task(memo_SHA) for memo_SHA in memo_SHA_list))

        self.memo_manager.memo_db['token_usage'] = tracker.summary()

        with open(project_root/"logs"/file_name, "w", encoding="utf-8") as f:
            json.dump(self.memo_manager.memo_db, f, ensure_ascii=False, indent=2)
        result_path = project_root / "logs" / file_name
        log.info(f"Results saved to: {result_path}")
        tracker.print_summary()
    



        
        




    



