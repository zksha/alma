import subprocess
import asyncio
from typing import Literal, Optional
from evals.logger import get_logger
import os

log = get_logger("main")

DOCKER_NAME = {
    'alfworld':'alfworld',
    'minihack':'balrog',
    'textworld':'balrog',
    'babaisai':'balrog',
    'babyai':'balrog'
}

BASELINES = {'g_memory', 'no_mem', 'similarity'}

async def run_evaluation(
                task_type:str, 
                mode: Literal["test","eval"], 
                memory_SHA: str, 
                eval_type = 'sequential', 
                model = 'gpt-4o-mini',
                train_size = 30,
                status = 'train',
                batch_max_update_concurrent = 10, 
                batch_max_retrieve_concurrent = 10,
                update_size: Optional[int]= None,
                update_task: Optional[str]= None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(script_dir)
    evals_dir = os.path.join(base_dir, "evals")
    memo_archive_dir = os.path.join(base_dir, "memo_archive")
    env_archive_dir = os.path.join(base_dir, "envs_archive")

    if memory_SHA in BASELINES:
        rel_dir = f'baseline/memo_structure_{memory_SHA}.py'
    else:
        rel_dir = f'{task_type}/memo_structure_{memory_SHA}.py'

    prefix = ""

    if DOCKER_NAME[task_type] == "alfworld":
        prefix = r"""
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate alfworld-env && \
        ALFWORLD_DATA_DIR="/opt/evals/data/alfworld" && \
        if [ ! -d "$ALFWORLD_DATA_DIR" ] || [ -z "$(ls -A "$ALFWORLD_DATA_DIR")" ]; then \
            echo "AlfWorld data not found. Downloading..." && \
            mkdir -p "$ALFWORLD_DATA_DIR" && \
            export ALFWORLD_DATA="$ALFWORLD_DATA_DIR" && \
            python /opt/evals/data/alfworld-download ; \
        fi && \
        """

    cmd = (
        "DOTENV_PATH=/opt/evals/.env "
        "mkdir -p /opt/evals/eval_envs && "
        f"cp /opt/envs_archive/{task_type}_envs.py /opt/evals/eval_envs/test_envs.py && "
        "mkdir -p /opt/evals/eval_envs/prompts && "
        f"cp /opt/envs_archive/prompts/{task_type}_prompt.py /opt/evals/eval_envs/prompts/env_prompt.py && "
        "mkdir -p /opt/evals/eval_envs/configs && "
        f"cp /opt/envs_archive/configs/{task_type}_config.yaml /opt/evals/eval_envs/configs/env_configs.yaml && "
        "mkdir -p /opt/evals/memo_test && "
        f"cp /opt/memo_archive/{rel_dir} /opt/evals/memo_test/memo_test.py && "
        f"{prefix} cd /opt/evals && "
        "stdbuf -oL -eL python -u -m launch "
        "--module_path memo_test/memo_test.py "
        f"--memory_id {memory_SHA} "
        f"--env {task_type} "
        f"--mode {mode} "
        f"--eval_type {eval_type} "
        f"--model {model} "
        f"--train_size {train_size} "
        f"--status {status} "
        f"--batch_max_update_concurrent {batch_max_update_concurrent} "
        f"--batch_max_retrieve_concurrent {batch_max_retrieve_concurrent} "
    )

    if update_size is not None:
        cmd += f"--update_size {update_size} "

    if update_task is not None:
        cmd += f"--update_task {update_task} "

    docker_cmd = [
        "docker", "run", "--rm", 
        "-it",
        "--mount", f"type=bind,source={evals_dir},target=/opt/evals",
        "--mount", f"type=bind,source={memo_archive_dir},target=/opt/memo_archive",
        "--mount", f"type=bind,source={env_archive_dir},target=/opt/envs_archive",
        "--mount", f"type=bind,source={base_dir}/.env,target=/opt/evals/.env",
        DOCKER_NAME[task_type],  # docker image name
        "bash", "-c", 
        f"{cmd}"
    ]

    process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=None,
            stderr=None
        )

    await process.wait()
    if process.returncode != 0:
        log.error(f"Container for memo {memory_SHA} exited with code {process.returncode}")
    else:
        log.info(f"✅ Container for memo {memory_SHA} finished")