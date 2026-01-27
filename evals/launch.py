from pathlib import Path
import asyncio
from agents.memo_structure import MemoStructure
from workflows.agent_workflow import Agent_Workflow
from eval_envs.base_envs import Basic_Recorder
from logger import get_logger
from typing import List, Dict, Optional
from collections import defaultdict
from agents.base import init_global_tracker
import random
import traceback
import json
import importlib
import argparse
import inspect
import numpy as np

log = get_logger("main", level_styles={
    "INFO": {"icon": "🚀", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})

LOG_DIR = Path("/opt/evals/logs")

def get_meta_eval_info(
    recorder_list: List[Basic_Recorder], 
    record_len: int,
    sample_size: int = 3, 
    eval_type: str = 'sequential',
):
    """
    Get information about the benchmark evaluation.
    Draw trajectories and calculate avg reward.
    Supports both sequential and batched evaluation types.
    """
    def process_records(records: List[Basic_Recorder], record_len: Optional[int] = None, tag: str = ""):
        rewards = []
        valid_records = []
        invalid_records_info = [f"[Update] Task failed with error: {repr(rec)}" for rec in records[record_len:]] # deal with updates error

        records = records[:record_len] 
        chunk_size = 3 if record_len else 1 # same as agent workflow
        each_chunk_size = record_len//chunk_size
        for chunk in range(chunk_size):
            chunk_rewards = []
            for idx in range(chunk*each_chunk_size,(chunk+1)*each_chunk_size):
                rec = records[idx]
                if isinstance(rec, Exception):
                    invalid_records_info.append(f"{tag} Task failed with error: {repr(rec)}")
                    chunk_rewards.append(0.0)
                    continue
                try:
                    chunk_rewards.append(rec.reward)
                    valid_records.append(rec)
                except Exception as e:
                    invalid_records_info.append(f"{tag} Record {rec} access error: {repr(e)}")
                    log.warning(f"{tag} Record {rec} access error: {repr(e)}")
                    chunk_rewards.append(0.0)
                    continue
            if chunk_rewards:
                rewards.append(chunk_rewards)

        group_avgs = [np.mean(group) for group in rewards]
        overall_avg = np.mean(group_avgs)
        overall_se = np.std(group_avgs, ddof=1) / np.sqrt(len(group_avgs)) if len(group_avgs) > 1 else 0.0

        reward_to_records: Dict[float, list] = defaultdict(list)
        reward_bins = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]
        for rec in valid_records:
            for i, (low, high) in enumerate(reward_bins):
                if low <= rec.reward < high or (i == len(reward_bins) - 1 and rec.reward == high):
                    reward_to_records[i].append(rec)
                    break
        sampled_records = []
        for r, recs in reward_to_records.items():
            k = min(sample_size, len(recs))
            sampled = random.sample(recs, k)
            sampled_records.extend(sampled)
            log.info(f"{tag}[Reward Layer {r} ({reward_bins[r][0]}-{reward_bins[r][1]})] drawn {k} trajectories")

        if invalid_records_info:
            invalid_sample_records = random.sample(
                invalid_records_info, min(sample_size, len(invalid_records_info))
            )
        else:
            invalid_sample_records = []

        return overall_avg, overall_se, sampled_records, invalid_sample_records, len(valid_records)

    # --- main logic ---
    retrieve_part = recorder_list
    avg_retrieve, std_retrieve, sample_retrieve, invalid_retrieve, valid_rec = process_records(retrieve_part, record_len, tag="[Retrieve]")

    sampled_records = sample_retrieve
    invalid_sample_records = invalid_retrieve

    log.info(f"--- AVG Retrieve Reward: {avg_retrieve:.3f} ---"
            f"--- SE Retrieve Reward: {std_retrieve:.3f} ---"
                f"--- Retrieve Valid: {valid_rec} ---"
                )
    eval_meta_info = {
        "benchmark_overall_eval_score": avg_retrieve,
        "benchmark_overall_eval_standard_deviation": std_retrieve
    }

    return eval_meta_info, sampled_records, invalid_sample_records

def get_json(env:str, file_name: str, sample_records: List[Basic_Recorder], avg_reward: Dict, invalid_sample_records: List[str], token_usg: Dict = {}):
    env_dir = LOG_DIR / env
    env_dir.mkdir(parents=True, exist_ok=True)
    
    examples = []
    for sample in sample_records:
        examples.append({
            "init_environment": sample.init,
            "memo_retrieved": sample.memory_retrieved,
            "steps": sample.steps,
            "final_reward": sample.reward
        })

    for error in invalid_sample_records:
        examples.append({
            "error_info": error,
            "final_reward": 0.0
        })

    episodes = {
        "benchmark_eval_score": avg_reward,
        "examples": examples,
        "token_usage": token_usg
        }
    
    file_path = env_dir / f"{file_name}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

    log.info(f"Benchmark Evaluation result saved to: {file_path}")
    return 

# def import_class_from_file(file_path: str, class_name: str):
#     """drawn class from specified path"""
#     spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
#     if spec is None:
#         raise ImportError(f"Cannot find spec for file {file_path}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return getattr(module, class_name)

def find_subclass_in_file(file_path: str, base_class: type):
    """find class name that inherits memo structure"""
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for file {file_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["dynamic_module"] = module  
    spec.loader.exec_module(module)

    subclasses = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            subclasses.append(obj)

    if not subclasses:
        raise ValueError(f"No class in {file_path} inherits from {base_class.__name__}")
    return subclasses[0]

async def main(module_path: str, 
                memory_id: str, 
                env: str, 
                batch_max_update_concurrent = 10, 
                batch_max_retrieve_concurrent = 10,
                mode = 'eval', 
                eval_type = 'seqential', 
                model = 'gpt-4o-mini',
                train_size = 500,
                status = 'train',
                update_size: Optional[int] = None,
                update_task: Optional[str] = None
                ):
    try: #initiate written memory structure
        memory_class = find_subclass_in_file(module_path, MemoStructure)
        if eval_type == 'batched':
            memory_structure = [memory_class()]
        else:
            memory_structure = [memory_class() for _ in range(3)]
    except Exception as e:
        log.warning(f'Fail to initiate memo strcture: {memory_id}')
        file_name = f"{memory_id}_{mode}"
        error_info = {"error_type": type(e).__name__, "message": str(e), "trace": traceback.format_exc()}
        get_json(env, file_name, [], {"benchmark_overall_eval_score": 0.0,"benchmark_overall_eval_standard_deviation": 0.0}, [error_info])
        return

    log.info(f"Start Evaluation for Memory Structure: {memory_id}")

    if update_task is None:
        update_task = status

    if update_size is not None:
        update_size = int(update_size)
    # log.info(update_size)

    tracker = init_global_tracker()

    agent_workflow = Agent_Workflow(
        memo_structure = memory_structure, 
        model = model,
        train_size = train_size,
        status = status,
        update_task = update_task
        )
    agent_workflow.init_env(env)

    record, record_len = await agent_workflow.run_all_tasks(mode = mode, 
                                                            batch_max_update_concurrent = batch_max_update_concurrent, 
                                                            batch_max_retrieve_concurrent = batch_max_retrieve_concurrent, 
                                                            eval_type = eval_type, 
                                                            update_size = update_size
                                                            )

    avg_reward, sampled_records, invalid_sample_records = get_meta_eval_info(
                                                            recorder_list = record, 
                                                            eval_type = eval_type, 
                                                            record_len = record_len
                                                            )

    token_tracker = tracker.summary()

    file_name = f"{memory_id}_{mode}"
    get_json(env, file_name, sampled_records, avg_reward, invalid_sample_records, token_tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", required=True, help="Path to memo structure")
    parser.add_argument("--memory_id", required=True, help="Memory hash value")
    parser.add_argument("--env", default="alfworld", help="task name")
    parser.add_argument("--batch_max_update_concurrent", default=10, help="number of update concurrent")
    parser.add_argument("--batch_max_retrieve_concurrent", default=10, help="number of retrieve concurrent")
    parser.add_argument("--mode", default='eval', help="mode for evals or execution test")
    parser.add_argument("--eval_type", default="seqential", help="can choose from \'sequential\' and \'batched\'")
    parser.add_argument("--model", default='gpt-4o-mini')
    parser.add_argument("--train_size", default=500)
    parser.add_argument("--status", default='train')
    parser.add_argument("--update_size", default=None)
    parser.add_argument("--update_task", default=None)
    
    args = parser.parse_args()
    # log.info(args.update_size)
    asyncio.run(main(module_path = args.module_path, 
                    memory_id = args.memory_id, 
                    env = args.env, 
                    batch_max_update_concurrent = int(args.batch_max_update_concurrent), 
                    batch_max_retrieve_concurrent = int(args.batch_max_retrieve_concurrent),
                    mode = args.mode, 
                    eval_type = args.eval_type, 
                    model = args.model, 
                    train_size = int(args.train_size), 
                    status = args.status, 
                    update_size = args.update_size, 
                    update_task = args.update_task
                    ))