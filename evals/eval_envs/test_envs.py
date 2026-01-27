from typing import Dict, Any, List, Optional
from typing_extensions import Literal
import asyncio
import numpy as np
from eval_envs.base_envs import Basic_Env, Basic_Recorder
import copy

from dataclasses import dataclass, field

from pathlib import Path
import yaml
import re
import subprocess
import random
import os
from collections import defaultdict
from logger import get_logger
import sys
from pathlib import Path
import sys, importlib, site

# find site packages
# site_packages = site.getsitepackages()
# for p in site_packages:
#     if p in sys.path:
#         sys.path.remove(p)
#     sys.path.insert(0, p)

# import textworld
try:
    import glob
    import gym
    textworld = importlib.import_module("textworld")
    textworld_gym = importlib.import_module("textworld.gym")
    from eval_envs.prompts.env_prompt import get_textworld_prompt
except Exception as e:
    print(e)
    pass


log = get_logger("main")



@dataclass
class TextworldRecorder(Basic_Recorder):
    """Recorder for a single environment interaction session."""

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Stores the initial observation. The actions may depend on tasks",
            "type": "Dict[str, Any]",
            "example":{
                'obs': '-= Spare Room =-\nYou arrive in a spare room. A normal kind of place.\n\n\n\nThere is a closed spherical portal leading south. There is a closed door leading...', 
                'goal': 'Please recover the latchkey from the floor of the pantry.'
                }
            }
    )

    steps: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "description": "List of all recorded steps in the session. We provide an example about the content that will be stored in steps(this is only a format sample, the content might not be the same).",
            "type": "List[Dict[str, Any]]",
            "example":[
                {
                'action_took': ['go south'], 
                'obs': '-= Basement =-\nYou arrive in a basement. A typical one.\n\nYou see a rack. The rack is typical. However, the rack, like an empty rack, has nothing on it. It would have been so cool if there was stuff on the rack.\n\nThere is an open hatch leading east. There is an open passageway leading north.', 
                'scores': 0.3314, 
                'dones': 'bool, indicating whether we finish the game(not neccessarily win)'
                }]
        }
    )

    reward: float = field(
        default=0.0,
        metadata={
            "description": "Final reward(progression) assigned to the session. From 0 to 1.",
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

    async def log_init(self, obs: Any, goal: Any, **kargs):
        async with self._lock:
            self.init = {"obs": obs, "goal": goal}

    async def log_memo_retrieved(self, memory_retrieved: Dict, **kargs):
        async with self._lock:
            self.memory_retrieved = memory_retrieved

    async def log_step(self, obs: Any, scores: Any, dones: Any, action_took: Any, **kargs):
        async with self._lock:
            info = {"action_took": action_took,  "obs": obs, "scores": scores, "dones": dones}
            self.steps.append(info)

    async def set_reward(self, reward: float, **kargs):
        async with self._lock:
            self.reward = reward


class Textworld_Env(Basic_Env):
    """Asynchronous textworld environment wrapper, returns a recorder at the end of episode."""

    def __init__(self,
        train_size, 
        train_eval: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "train",
        update_task: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution",
        max_trails: int = 80,
        **kwargs
        ):   

        self.project_root = Path(__file__).resolve().parent.parent
        self.textworld_games_path = self.project_root.parent/'balrog'/'tw_games'
        self.max_trails = max_trails
        config_path = self.project_root / 'eval_envs' / 'configs' / 'env_configs.yaml'
        self.textworld_kwargs = from_yaml(config_path)
        required_kwargs = ["objective", "description", "score", "max_score", "won"]
        for kwarg in required_kwargs:
            assert kwarg in self.textworld_kwargs
        
        self.request_infos = textworld.EnvInfos(**self.textworld_kwargs)

        # register envs
        self.category_to_tasks = defaultdict(list)
        
        for pattern in ["*.ulx", "*.z8"]:
            for entry in sorted(glob.glob(os.path.join(self.textworld_games_path, f"**/{pattern}"), recursive=True)):
                task = Path(entry).parent.name
                if task in ['treasure_hunter','the_cooking_game']:
                    env_id = textworld.gym.register_game(entry, self.request_infos, max_episode_steps=self.max_trails+10)
                    self.category_to_tasks[task].append(env_id)

        random.seed(42)
        ood_cats = []

        train_tasks = []
        test_in_domain = []
        test_out_of_domain = []

        for cat, cat_tasks in self.category_to_tasks.items():
            if cat in ood_cats:
                test_out_of_domain += cat_tasks
            else:
                random.shuffle(cat_tasks)
                n = len(cat_tasks)
                train_n = int(n * 0.5)
                train_tasks += cat_tasks[:train_n]
                test_in_domain += cat_tasks[train_n:]
        
        if train_eval == 'train':
            self.task_list = train_tasks[:min(len(train_tasks), train_size)]
        elif train_eval == 'eval_in_distribution':
            self.task_list = test_in_domain
        else:
            self.task_list = test_out_of_domain

        if update_task == 'train':
            self.update_task_list = train_tasks[:min(len(train_tasks), train_size)]
        elif update_task == 'eval_in_distribution':
            self.update_task_list = test_in_domain
        else:
            self.update_task_list = test_out_of_domain
        
        self.recorders = {}
        self.envs = {}


    async def set_task_env(self, task: str) -> Any:
        """Initialize environment for a task and log initial observation."""
    
        env = textworld.gym.make(task.rsplit("_runtime_", 1)[0])
        env = TextWorldWrapper(env, max_steps=self.max_trails)

        textworld_recorder = TextworldRecorder()

        obs = env.reset()
        
        await textworld_recorder.log_init(**obs)

        self.recorders[task] = textworld_recorder
        self.envs[task] = env

        return obs, textworld_recorder

    async def run_step(self, action: str, game_files: str, **kargs) -> Dict:
        """Run one step and log results."""
        if len(action) >= 256:
            raise RuntimeError(f"Action: {action} is not valid.")
        obs, reward, done, info = self.envs[game_files].step(action)
        # print(obs['text'], reward,done)
        obs['action_took'] = action
        obs['scores'] = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)
        obs['dones'] = done
        
        recorder = self.recorders[game_files]
        await recorder.log_step(**obs)
        return  obs, recorder

    async def cal_reward(self, scores: float, game_files: str) -> TextworldRecorder:
        """Store a simple binary reward and return recorder."""
        recorder = self.recorders[game_files]
        self.envs[game_files].close() # avoid OOM
        await recorder.set_reward(scores)
        return recorder

    async def get_prompt(self, game_files: str, obs: str, memory_retrived: Dict = {}, goal = '', **kargs):
        task_type = [k for k, v in self.category_to_tasks.items() if game_files.rsplit("_runtime_", 1)[0] in v][0]
        
        prompt = get_textworld_prompt(
            task_type = task_type,
            obs=obs,
            memory_retrived=memory_retrived,
            **kargs)
        return prompt
        
def from_yaml(yaml_path: str):
    """Load record_db from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config

class AlwaysTrue:
    def __contains__(self, item):
        return True

class TextWorldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=40):
        super().__init__(env)
        self.language_action_space = AlwaysTrue()
        self.progression = 0.0
        self.max_steps = max_steps
        self.action_space = gym.spaces.Space()
        self.observation_space = gym.spaces.Space()

    @property
    def default_action(self):
        return "help"

    def filter_objective(self, obs, info):
        objective = info["objective"]
        parts = obs.split(objective)
        if len(parts) == 1:
            return parts[0].strip()
        else:
            return parts[-1].strip()

    def reset(self):
        obs, info = self.env.reset()
        obs = self.filter_objective(obs, info)
        self.progression = 0.0
        return {"obs": obs, 'goal': info["objective"]}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.filter_objective(obs, info)

        if done:
            self.progression = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)

        return {"obs": obs}, reward, done, info

    def get_stats(self):
        return {"progression": self.progression}