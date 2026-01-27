from dataclasses import dataclass
from typing import Optional, List
import traceback
from utils.hire_agent import Agent
from agents.memo_structure import MemoStructure
import eval_envs.test_envs as env_module
import asyncio
import copy
import random
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import sys
from logger import get_logger
log = get_logger("main")

ENVS = {
    'alfworld': 'ALFWorld_Env',
    'minihack': 'MiniHack_Env',
    'textworld': 'Textworld_Env',
    'babaisai': 'Babaisai_Env',
    'babyai': 'Babyai_Env'
}

@dataclass
class Agent_Workflow:
    def __init__(self, memo_structure: List[MemoStructure], model: str, train_size = 500, status = 'train', update_task = 'train'):
        self.memo = memo_structure
        self.reasoning_effort = None
        if '/' in model:
            self.model = model.split('/')[0]
            self.reasoning_effort = model.split('/')[1]
        else:
            self.model = model
        self.status = status
        self.update_task = update_task
        self.train_size = train_size
        self.update_task = update_task
        self.env_args = {
            'train_eval': status,
            'train_size': train_size,
            'update_task': update_task
        }
    
    def init_env(self, env_type: str):
        assert env_type in ENVS
        self.env_type = env_type
        env_class = getattr(env_module, ENVS[self.env_type])
        self.env = env_class(**self.env_args)
        self.task_list = self.env.task_list
        return self.task_list
    
    async def run_single_task(self, task_path: str, eval_type = 'sequential', current_state = 'update', progress = None, task_id = None, memory_snapshot = None):
        
        agent = Agent(
            model = self.model,
            system_prompt = ""
            )
        present_path = task_path
        task_env, recorder = await self.env.set_task_env(task_path)
        if memory_snapshot is None:
            memory_snapshot = self.memo[0] #utilize newest memory archive if no snapshot provided
        
        if eval_type == 'sequential' or (eval_type == 'batched' and current_state == 'retrieve'):
            try:
                retrieved_memo = await asyncio.wait_for(memory_snapshot.general_retrieve(recorder), timeout=300)
            except Exception as e:
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"[red]Retrieve failed for {present_path}: {e}...[/red]"
                    )
                print(f"[red]Retrieve failed for {present_path}: {e}...[/red]")
                raise
        else:
            retrieved_memo = {}

        task_env['memory_retrived'] = retrieved_memo
        
        await self.env.recorders[task_path].log_memo_retrieved(retrieved_memo)

        prompts = await self.env.get_prompt(game_files=task_path, **task_env)
        
        agent.messages[0]['content'] = prompts[0]['content'] + agent.messages[0]['content']
        user_prompt = prompts[1]['content']
        new_env = {'dones':False}

        for steps in range(self.env.max_trails):
            try:
                if self.reasoning_effort is not None:
                    answer = await asyncio.wait_for(agent.ask(user_prompt, with_history = True, reasoning_effort = self.reasoning_effort), timeout=300)
                else:
                    answer = await asyncio.wait_for(agent.ask(user_prompt, with_history = True), timeout=300)
                if answer is None:
                    raise RuntimeError('Previous Action is not recognized or not supported for this environment')
                new_env, recorder = await self.env.run_step(answer.strip(), task_path, **new_env)
                if new_env['dones']:
                    break
                user_prompt = await self.env.get_prompt(game_files = task_path,**new_env)
                user_prompt = user_prompt[1]['content']
                # print(user_prompt)
                # print(len(user_prompt))
            except Exception as e:
                err_msg = traceback.format_exc()
                # print(f"[ERROR]: {e}")
                if "context length" in str(e).lower() or "maximum context length" in str(e).lower():
                    raise ValueError("Prompt exceeded model context limit. ")
                # log.info(e)
                # print(e)
                error_feedback = f""" An error occurred in the last step:
                {err_msg}
                Please revise your next action to avoid this issue. 
                Respond strictly with a valid action from the available action list.
                """
                user_prompt = user_prompt + "\n" + error_feedback
            finally:
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

        # Calculate and update rewards
        recorder = await self.env.cal_reward(new_env['scores'], task_path)
        if eval_type == 'sequential': # for sequential perform updates seperately, for batched perform update after finish all tasks in concurrent.
            try:
                await memory_snapshot.general_update(recorder)
            except Exception as e:
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"[red]Update failed for {present_path}: {e}...[/red]"
                    )
                print(f"[red]Update failed for {present_path}: {e}...[/red]")
                raise

        if hasattr(recorder, "reward"):
            progress.update(task_id, description=f"[{'green' if recorder.reward == 1 else 'white'}]---Finish Task: {present_path} in {self.env_type} | Reward: {recorder.reward} | Steps Take: {steps+1}---[/{'green' if recorder.reward == 1 else 'white'}]")
        return recorder


    async def run_all_tasks(self, mode = 'eval', batch_max_update_concurrent = 5, batch_max_retrieve_concurrent = 5, eval_type = 'sequential', update_size: Optional[int] = None):
        results = []
        task_list = self.task_list.copy()
        if mode != 'eval':
            task_list = random.sample(task_list, min(len(task_list),5))
        
        async def sem_task(task_path, eval_type, semaphore, current_state = 'update', task_id = 0, progress = None, memory_snapshot = None):
            async with semaphore:
                try:
                    result = await self.run_single_task(task_path, eval_type = eval_type, current_state = current_state, progress = progress, task_id=task_id, memory_snapshot = memory_snapshot)
                except Exception as e:
                    result = e
                    print(f"[ERROR]Task {task_path} failed: {e}")
                return result
        
        random.seed(42)
        random.shuffle(task_list)
        mid = len(task_list)//2
        if self.update_task == self.status:
            if update_size is not None: # for evaluation of different update size
                assert update_size <= mid
                update_task = task_list[:update_size]
            else:
                update_task = task_list[:mid]
        else:
            if update_size is not None: # for evaluation of different update size
                assert update_size <= len(self.env.update_task_list)
                # print(self.env.update_task_list)
                # print(self.env.task_list)
                update_task = self.env.update_task_list[:update_size]
            else:
                update_task = self.env.update_task_list
        # log.info(update_size)
        log.info(f'Update Dataset: {self.update_task}, Update data size: {len(update_task)}. Perform update...')
        with Progress(TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                style="grey35",           
                complete_style="deep_sky_blue3",  
                finished_style="deep_sky_blue3"     
            ),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=False) as update_progress:
            task_ids = {}
            for i, task_path in enumerate(update_task):
                task_ids[task_path] = update_progress.add_task(task_path, total=self.env.max_trails, position=i)
                update_progress.start_task(task_ids[task_path])
            semaphore = asyncio.Semaphore(batch_max_update_concurrent)
            coroutines = [sem_task(task_path, 'batched', current_state = 'update', task_id=task_ids[task_path], progress=update_progress, semaphore = semaphore) for task_path in update_task]
            update_results = await asyncio.gather(*coroutines)
            update_results_errors = [result for result in update_results if isinstance(result, Exception)]
            update_results = [result for result in update_results if not isinstance(result, Exception)]
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                style="grey35",
                complete_style="deep_sky_blue3",
                finished_style="deep_sky_blue3"
            ),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=False
        ) as update_progress:

            task = update_progress.add_task("Updating recorders", total=len(update_results))
            for recorder in update_results:
                try:
                    if eval_type == 'batched':
                        await self.memo[0].general_update(recorder)
                    else:
                        for memo_structure in self.memo: # do update multiple times for sequential process
                            await memo_structure.general_update(recorder)
                except Exception as e:
                    print(f"[red]Update failed: {e}...[/red]")
                    update_results_errors.append(e)
                finally:
                    update_progress.update(task, advance=1)
        if eval_type == 'sequential':

            async def run_group(snapshot, tasks, task_ids, progress):
                # sequential for each group
                semaphore = asyncio.Semaphore(1)
                group_results = []
                for i, task_path in enumerate(tasks, start=1):
                    tid = task_ids[task_path]
                    result = await sem_task(
                                            task_path, 
                                            eval_type, 
                                            task_id=tid,
                                            progress = progress, 
                                            semaphore = semaphore, 
                                            memory_snapshot = snapshot
                                            )
                    group_results.append(result)
                return group_results
            
            retrieve_task = []
            for i in range(3):
                temp = [f"{task_path}_runtime_{i+1}" for task_path in task_list[mid:]]
                retrieve_task.append(temp)
            log.info(f'Retrieve Dataset: {self.status}, Retrieve data size: {len(retrieve_task)*len(retrieve_task[0])}. Perform retrieve...')
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None, style="grey35", complete_style="deep_sky_blue3", finished_style="deep_sky_blue3"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=False
            ) as progress:
                task_ids = {}
                for group_idx, group_tasks in enumerate(retrieve_task):
                    for i, task_path in enumerate(group_tasks):
                        tid = progress.add_task(task_path, total=self.env.max_trails, position=i + group_idx*len(group_tasks))
                        task_ids[task_path] = tid

                results = await asyncio.gather(*[
                    run_group(self.memo[i], retrieve_task[i], task_ids, progress) for i in range(3)
                ])
                flat_results = [item for group in results for item in group]
                return flat_results+update_results_errors, len(flat_results)
        else:
            retrieve_task = []
            for i in range(3):  
                for task_path in task_list[mid:]:  
                    retrieve_task.append(f"{task_path}_runtime_{i+1}")
            # retrieve_task = task_list[mid:]*3
            log.info(f'Retrieve Dataset: {self.status}, Retrieve data size: {len(retrieve_task)}. Perform retrieve...')
            with Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="grey35",           
                    complete_style="deep_sky_blue3",  
                    finished_style="deep_sky_blue3"     
                ),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=False) as retrieve_progress:
                log.info('perform retrieve..')
                for i, task_path in enumerate(retrieve_task):
                    # orig_task_path, repeat_idx_str = task_path.rsplit("_runtime_", 1)
                    task_ids[task_path] = retrieve_progress.add_task(task_path, total=self.env.max_trails, position=i)
                    retrieve_progress.start_task(task_ids[task_path])
                semaphore = asyncio.Semaphore(batch_max_retrieve_concurrent)
                coroutines = [sem_task(task_path, eval_type, current_state = 'retrieve', task_id=task_ids[task_path], progress=retrieve_progress, semaphore = semaphore) for task_path in retrieve_task]
                retrieve_results = await asyncio.gather(*coroutines)
        # print('return retrieve results')
        return retrieve_results+update_results_errors, len(retrieve_results)

    


    









        
        

