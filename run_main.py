import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional
# add evals folder to sys.path
evals_path = Path(__file__).resolve().parent / "evals"
sys.path.insert(0, str(evals_path))
import gc
gc.collect()
from core.meta_agent import MetaAgent

async def main(args):
    meta_agent = MetaAgent(
        args.task_type, 
        args.meta_model,
        args.execution_model,
        args.status,
        args.history_ckpt_path
        )
    if args.status == 'train':
        await meta_agent.forward(
            rollout_type = args.rollout_type,
            steps = args.steps,
            max_concurrent = args.max_container_concurrent,
            result_dir = args.result_dir,
            batch_max_update_concurrent = args.batch_max_update_concurrent,
            batch_max_retrieve_concurrent = args.batch_max_retrieve_concurrent,
            update_size = args.update_size,
            train_size = args.train_size
            )
    else:
        await meta_agent.run_single_memo(
            memo_SHA = args.memo_SHA,
            eval_type = args.rollout_type,
            status = args.status,
            batch_max_update_concurrent = args.batch_max_update_concurrent,
            batch_max_retrieve_concurrent = args.batch_max_retrieve_concurrent,
            update_size = args.update_size,
            update_task = args.update_task
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="alfworld")
    parser.add_argument("--rollout_type", type=str, default="sequential",
                        choices=["sequential", "batched"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_container_concurrent", type=int, default=5)
    parser.add_argument("--result_dir", type=str, default="check")
    parser.add_argument("--meta_model", type=str, default="gpt-4.1")
    parser.add_argument("--execution_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--memo_SHA", type=str, default=None)
    parser.add_argument("--status", type=str, default='train')
    parser.add_argument("--batch_max_update_concurrent", type=int, default=10)
    parser.add_argument("--batch_max_retrieve_concurrent", type=int, default=10)
    parser.add_argument("--update_size", type=int, default=None)
    parser.add_argument("--update_task", type=str, default=None)
    parser.add_argument("--train_size", type=str, default=500)
    parser.add_argument("--history_ckpt_path", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main(args))
