from typing import Dict, Any, List
from typing_extensions import Literal
import numpy as np
from eval_envs.base_envs import Basic_Env, Basic_Recorder
import copy
import asyncio

from dataclasses import dataclass, field

from pathlib import Path
import yaml
import re
import subprocess
import os
from logger import get_logger

try:
    from alfworld.agents.environment import get_environment
    from eval_envs.prompts.env_prompt import get_alfworld_prompt
except Exception as e:
    pass


log = get_logger("main")
TASK_MAP = {
    "train": 'train',
    "eval_in_distribution": 'valid_seen',
    "eval_out_of_distribution" : 'valid_unseen'
}

@dataclass
class ALFWorldRecorder(Basic_Recorder):
    """Recorder for a single environment interaction session."""

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Stores the initial observation and info when environment is set. ",
            "type": "Dict[str, Any]",
            "example":{
                'obs': '-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 17, a drawer 16, a drawer 15, a drawer 14, a drawer 13, a drawer 12, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.\n\nYour task is to: look at alarmclock under the desklamp.', 
                'actions_list': ['go to bed 1', 'go to desk 1', 'go to drawer 1', 'go to drawer 10', 'go to drawer 11', 'go to drawer 12', 'go to drawer 13', 'go to drawer 14', 'go to drawer 15', 'go to drawer 16', 'go to drawer 17', 'go to drawer 2', 'go to drawer 3', 'go to drawer 4', 'go to drawer 5', 'go to drawer 6', 'go to drawer 7', 'go to drawer 8', 'go to drawer 9', 'go to dresser 1', 'go to garbagecan 1', 'go to shelf 1', 'go to shelf 2', 'go to shelf 3', 'go to shelf 4', 'go to shelf 5', 'go to shelf 6', 'help', 'inventory', 'look']
                }
            }
    )

    steps: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "description": "List of all recorded steps in the session.",
            "type": "List[Dict[str, Any]]",
            "example":[{
                'action_took': ['go to desk 1'], 
                'obs': 'You arrive at desk 1. On the desk 1, you see a alarmclock 1, a bowl 1, a cd 2, a cd 1, a mug 2, a mug 1, and a pencil 1.', 
                'scores': 0, 
                'dones': False, 
                'actions_list': ['examine desk 1', 'go to bed 1', 'go to drawer 1', 'go to drawer 10', 'go to drawer 11', 'go to drawer 12', 'go to drawer 13', 'go to drawer 14', 'go to drawer 15', 'go to drawer 16', 'go to drawer 17', 'go to drawer 2', 'go to drawer 3', 'go to drawer 4', 'go to drawer 5', 'go to drawer 6', 'go to drawer 7', 'go to drawer 8', 'go to drawer 9', 'go to dresser 1', 'go to garbagecan 1', 'go to shelf 1', 'go to shelf 2', 'go to shelf 3', 'go to shelf 4', 'go to shelf 5', 'go to shelf 6', 'help', 'inventory', 'look', 'take alarmclock 1 from desk 1', 'take bowl 1 from desk 1', 'take cd 1 from desk 1', 'take cd 2 from desk 1', 'take mug 1 from desk 1', 'take mug 2 from desk 1', 'take pencil 1 from desk 1']
                }]
        }
    )

    reward: float = field(
        default=0.0,
        metadata={
            "description": "Final reward assigned to the session.",
            "type": "float"
        }
    )

    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        metadata={
            "description": "Async lock to make log operations thread-safe.",
            "type": "asyncio.Lock",
            "internal": True
        }
    )

    memory_retrieved: Dict = field(
        default_factory=dict)

    async def log_init(self, obs: Any, actions_list: Any):
        async with self._lock:
            self.init = {"obs": obs, "actions_list": actions_list}

    async def log_memo_retrieved(self, memory_retrieved: Dict):
        async with self._lock:
            self.memory_retrieved = memory_retrieved

    async def log_step(self, action_took: str, obs: Any, scores: Any, dones: Any, actions_list: Any):
        async with self._lock:
            self.steps.append(
                {"action_took": action_took, "obs": obs, "scores": scores, "dones": dones, "actions_list": actions_list}
            )

    async def set_reward(self, reward: float):
        async with self._lock:
            self.reward = reward


class ALFWorld_Env(Basic_Env):
    """Asynchronous ALFWorld environment wrapper, returns a recorder at the end of episode."""

    def __init__(
        self,
        max_trails: int = 30,
        train_eval: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "train",
        train_size: int = 10,
        update_task: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution",
        **kargs
    ):     

        #check for env path
        self.project_root = Path(__file__).resolve().parent.parent
        default_data_path = self.project_root/"data/alfworld"
        current_env = os.environ.get("ALFWORLD_DATA")

        if current_env != str(default_data_path):
            log.warning(f"Setting $ALFWORLD_DATA to {default_data_path}")
            os.environ["ALFWORLD_DATA"] = str(default_data_path)

        config_path = self.project_root / 'eval_envs' / 'configs' / 'env_configs.yaml'
        config = from_yaml(config_path)

        self.env_config = config
        self.max_trails = max_trails
        self.train_eval = train_eval
        self.train_size = train_size
        self.update_task_list = self.get_files(os.path.join(self.project_root/"data/alfworld/json_2.1.1", TASK_MAP[update_task].lstrip("/")))
        self.task_list = self.get_task()

        try:
            self.env_template = get_environment(self.env_config["env"]["type"])(
                self.env_config, train_eval=self.train_eval
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ALFWorld environment template: {e}")

        self.recorders = {}
        self.envs = {}

    def get_files(self, root_path):
        pddl_files = []
        for game_dir in os.listdir(root_path):
            game_path = os.path.join(root_path, game_dir)
            if os.path.isdir(game_path):
                for trial_dir in os.listdir(game_path):
                    trial_path = os.path.join(game_path, trial_dir)
                    if os.path.isdir(trial_path):
                        for file_name in os.listdir(trial_path):
                            if file_name.endswith(".tw-pddl"):
                                full_path = os.path.join(trial_path, file_name)
                                pddl_files.append(full_path)
        return pddl_files

    def get_task(self):
        """
        get task file paths as list for task running
        """        
        if self.train_eval == 'train':
            root_path = os.path.join(self.project_root/"data/alfworld/json_2.1.1", TASK_MAP[self.train_eval].lstrip("/"))
            pddl_files = self.get_files(root_path)
            pddl_files = pddl_files[:self.train_size]
        elif self.train_eval == 'eval_in_distribution':
            # train_root_path = os.path.join(self.project_root/"data/alfworld/json_2.1.1", TASK_MAP['train'].lstrip("/"))
            # train_pddl_files = get_files(train_root_path)
            # train_pddl_files = train_pddl_files[self.train_size:]
            root_path = os.path.join(self.project_root/"data/alfworld/json_2.1.1", TASK_MAP[self.train_eval].lstrip("/"))
            pddl_files = self.get_files(root_path)
            # pddl_files += train_pddl_files
            print(f'{self.train_eval} on {len(pddl_files)} games')
        else:
            root_path = os.path.join(self.project_root/"data/alfworld/json_2.1.1", TASK_MAP[self.train_eval].lstrip("/"))
            pddl_files = self.get_files(root_path)
        return pddl_files

    def _clone_env(self):
        """Create a fresh environment instance from template."""
        return copy.deepcopy(self.env_template)

    async def set_task_env(self, game_files: str) -> Any:
        """Initialize environment for a task and log initial observation."""
        env = self._clone_env()
        env.game_files = [game_files.rsplit("_runtime_", 1)[0]]
        env = env.init_env(batch_size=1)

        alf_recorder = ALFWorldRecorder()

        obs, info = env.reset()

        await alf_recorder.log_init(obs[0], info['admissible_commands'][0])

        self.recorders[game_files] = alf_recorder
        self.envs[game_files] = env

        return {'obs': obs[0], 'actions_list': info['admissible_commands'][0]}, alf_recorder

    async def run_step(self, action: str, game_files: str, **kargs) -> Dict:
        """Run one step and log results."""
        obs, scores, dones, infos = self.envs[game_files].step([action])

        recorder = self.recorders[game_files]
        await recorder.log_step(action, obs[0], scores[0], dones[0], infos['admissible_commands'][0])
        return  {"action_took": action, "obs": obs[0], "scores": scores[0], "dones": dones[0], "actions_list": infos['admissible_commands'][0]}, recorder

    async def cal_reward(self, scores: float, game_files: str) -> ALFWorldRecorder:
        """Store a simple binary reward and return recorder."""
        recorder = self.recorders[game_files]
        await recorder.set_reward(scores)
        return recorder

    async def get_prompt(self, obs: str, actions_list: List[str], game_files: str, memory_retrived: Dict = {}, **kargs):
        return get_alfworld_prompt(obs, actions_list, memory_retrived, **kargs)
        
def from_yaml(yaml_path: str):
    """Load record_db from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config
