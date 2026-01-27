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

try:
    from balrog.environments.nle import AutoMore, NLELanguageWrapper
    import gym
    from eval_envs.prompts.env_prompt import get_minihack_prompt, get_available_actions
except Exception as e:
    pass
import sys, importlib, site

# find site packages
site_packages = site.getsitepackages()
for p in site_packages:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
try:
    minihack = importlib.import_module("minihack")
except Exception as e:
    pass


log = get_logger("main")



@dataclass
class MiniHackRecorder(Basic_Recorder):
    """Recorder for a single environment interaction session."""

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Stores the initial observation(including long-term and short-term observation) and actions with explaination. The possible actions will be the same through out the whole game(but maybe different for different games).",
            "type": "Dict[str, Any]",
            "example":{
                'long_term_context': 'message:\nHello Agent, welcome to NetHack!  You are a chaotic female human Monk.\n\nlanguage observation:\nbars far eastnortheast and eastsoutheast\nbars far eastnortheast and eastsoutheast\ndark area far east, southeast, and south\nbars near north, northnortheast, northeast, eastnortheast, east, eastsoutheast, southeast, southsoutheast, southsouthwest, westsouthwest, westnorthwest, and northnorthwest\nbars near north, northnortheast, northeast, eastnortheast, east, eastsoutheast, southeast, southsoutheast, southsouthwest, westsouthwest, westnorthwest, and northnorthwest\ndark area near north, northeast, southwest, west, and northwest\nboulder near southsoutheast\nfountain near southsoutheast and south\nfountains near southsouthwest\nbars very near north, northnortheast, northeast, eastnortheast, east, eastsoutheast, southeast, southwest, westsouthwest, west, westnorthwest, northwest, and northnorthwest\nboulder very near southsouthwest\nbars adjacent north, northeast, west, and northwest\nboulder adjacent southeast and south\n\ncursor:\nYourself a monk\n(x=37, y=12)\n\nmap:\n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                  ##########                                    \n                                  ##########                                    \n                                  ##########                                    \n                                  ##########                                    \n                                  ###@.#####                                    \n                                  ##.``#####                                    \n                                  ##`..#####                                    \n                                  #{.{....##                                    \n                                  #.{.`{.###                                    \n                                  ##########                                    \n                                                                                \n                                                                                \n                                                                                \n                                                                                \nAgent the Candidate            St:18/02 Dx:11 Co:10 In:14 Wi:15 Ch:9 Chaotic S: \nDlvl:1 $:0 HP:14(14) Pw:4(4) AC:4 Xp:1/0                                        \n\n', 
                'short_term_context': 'inventory:\na: an uncursed +2 pair of leather gloves (being worn)\nb: an uncursed +1 robe (being worn)\nc: a blessed spellbook of protection\nd: a blessed scroll of identify\ne: 3 uncursed potions of healing\nf: 4 uncursed food rations\ng: 6 uncursed apples\nh: 6 uncursed oranges\ni: 3 uncursed fortune cookies\n',
                'action_dict': {
                    "north": "move north",
                    "east": "move east",
                    "south": "move south",
                    "west": "move west",
                    "northeast": "move northeast",
                    "southeast": "move southeast",
                    "southwest": "move southwest",
                    "northwest": "move northwest",
                    "far north": "move far north",
                    "far east": "move far east",
                    "far south": "move far south",
                    "far west": "move far west"
                    },
                'goal': 'You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them.'
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
                'action_took': ['north'], 
                'long_term_context': 'message:\nYou cannot pass through the bars.\n\nlanguage observation:\nbars far eastnortheast, east, eastsoutheast, southeast, southsoutheast, south, and southsouthwest\ndark area far east, southeast, and south\nfountain far southsoutheast\nbars near eastnortheast, east, eastsoutheast, and southsouthwest\nbars near eastnortheast, east, eastsoutheast, and southsouthwest\nfountain near southsoutheast and south\nboulders near southsoutheast\ndark area very near north, northeast, southwest, west, and northwest\nbars very near eastnortheast, east, eastsoutheast, and southsouthwest\nfountain very near southsoutheast\nbars adjacent north, northeast, southwest, west, and northwest\nboulder adjacent southeast and south\n\ncursor:\nYourself a priestess\n(x=35, y=9)\n\nmap:\n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                  ##########                                    \n                                  #@.#######                                    \n                                  #``#######                                    \n                                  #.{.######                                    \n                                  #...######                                    \n                                  #{..{..###                                    \n                                  ###.``.###                                    \n                                  #####...##                                    \n                                  ######{.##                                    \n                                  ##########                                    \n                                                                                \n                                                                                \n                                                                                \n                                                                                \nAgent the Aspirant             St:11 Dx:16 Co:16 In:9 Wi:16 Ch:11 Chaotic S:0   \nDlvl:1 $:0 HP:13(13) Pw:9(9) AC:7 Xp:1/0                                        \n\n', 
                'short_term_context': 'inventory:\na: a blessed +1 mace (weapon in hand)\nb: a +0 robe (being worn)\nc: a blessed +0 small shield (being worn)\nd: 4 potions of holy water\ne: 2 cloves of garlic\nf: a sprig of wolfsbane\ng: a spellbook of identify\nh: a spellbook of create monster\ni: a tooled horn\n',
                'scores': -0.001, 
                'dones': False
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

    async def log_init(self, long_term_context: Any, short_term_context: Any, action_dict: Any, goal: Any, **kargs):
        async with self._lock:
            self.init = {"long_term_context": long_term_context, "short_term_context": short_term_context, "action_dict": action_dict, "goal": goal}

    async def log_memo_retrieved(self, memory_retrieved: Dict, **kargs):
        async with self._lock:
            self.memory_retrieved = memory_retrieved

    async def log_step(self, long_term_context: Any, short_term_context: Any, scores: Any, dones: Any, action_took: Any, **kargs):
        async with self._lock:
            info = {"action_took": action_took, "long_term_context": long_term_context, "short_term_context": short_term_context, "scores": scores, "dones": dones}
            self.steps.append(info)

    async def set_reward(self, reward: float, **kargs):
        async with self._lock:
            self.reward = reward


class MiniHack_Env(Basic_Env):
    """Asynchronous minihack environment wrapper, returns a recorder at the end of episode."""

    def __init__(self,
        train_size, 
        train_eval: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "train",
        update_task: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution",
        max_trails: int = 100,
        **kargs
        ):   

        self.project_root = Path(__file__).resolve().parent.parent
        self.task_list = []
        config_path = self.project_root / 'eval_envs' / 'configs' / 'env_configs.yaml'
        self.minihack_kwargs = from_yaml(config_path)
        self.max_trails = max_trails


        # print(gym.envs.registry.all())
        for env_spec in gym.envs.registry.all():
            id = env_spec.id
            # print(id)
            exclude = ("Custom", "MultiRoom", "LavaCrossing", "SimpleCrossing")
            if id.startswith("MiniHack") and all(x not in id for x in exclude):
                self.task_list.append(id)
        
        def task_category(task_name):
            if "corridor" in task_name.lower():
                return "corridor"
            elif "quest" in task_name.lower():
                return "quest"
            elif "boxoban" in task_name.lower():
                return "boxoban"
            else:
                return "distance"

        category_to_tasks = defaultdict(list)
        for t in self.task_list:
            category_to_tasks[task_category(t)].append(t)

        random.seed(42)
        # for key, val in category_to_tasks.items():
        #     print(f"{key}: {len(val)}")
        categories = list(category_to_tasks.keys())
        ood_cats = ["boxoban","corridor","quest"]

        # def task_category(task_name):
        #     return "-".join(task_name.split("-")[:2])

        # category_to_tasks = defaultdict(list)
        # for t in self.task_list:
        #     category_to_tasks[task_category(t)].append(t)

        # random.seed(42) #org random state 42 ， current 123
        # # print(category_to_tasks)
        # categories = list(category_to_tasks.keys())
        # ood_cats = random.sample(categories, 3)

        train_tasks = []
        test_in_domain = []
        test_out_of_domain = []

        for cat, cat_tasks in category_to_tasks.items():
            if cat in ood_cats:
                test_out_of_domain += cat_tasks
            else:
                random.shuffle(cat_tasks)
                n = len(cat_tasks)
                train_n = int(n * 0.3) #original 0.2, current 0.3
                train_tasks += cat_tasks[:train_n]
                test_in_domain += cat_tasks[train_n:]
        
        if train_eval == 'train':
            self.task_list = train_tasks[:min(len(train_tasks), train_size)]
            
        elif train_eval == 'eval_in_distribution':
            self.task_list = test_in_domain
            # self.task_list = ['MiniHack-Sokoban4a-v0']
        else:
            self.task_list = test_out_of_domain

        if update_task == 'train':
            self.update_task_list = train_tasks[:min(len(train_tasks), train_size)]
        elif update_task == 'eval_in_distribution':
            self.update_task_list = test_in_domain
        else:
            self.update_task_list = test_out_of_domain
        
        self.skip_more = self.minihack_kwargs.pop("skip_more", False)
        self.recorders = {}
        self.envs = {}


    async def set_task_env(self, task: str) -> Any:
        """Initialize environment for a task and log initial observation."""
        env = gym.make(
            task.rsplit("_runtime_", 1)[0],
            observation_keys=[
                "glyphs",
                "blstats",
                "tty_chars",
                "inv_letters",
                "inv_strs",
                "tty_cursor",
                "tty_colors",
            ],**self.minihack_kwargs
        )
        if self.skip_more:
            env = AutoMore(env)
        env = NLELanguageWrapper(env)

        minihack_recorder = MiniHackRecorder()

        obs = env.reset()
        obs = obs['text']
        
        action_dict = get_available_actions(env)

        obs['action_dict'] = action_dict
        if "corridor" in task.lower():
            goal = "Your goal is to explore the level and reach the stairs down"
        elif "quest" in task.lower():
            goal = "Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs down."
        elif "boxoban" in task.lower():
            goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
        else:
            goal = "Your goal is to get as far as possible in the game."
        
        obs['goal'] = goal

        await minihack_recorder.log_init(**obs)

        self.recorders[task] = minihack_recorder
        self.envs[task] = env
        return obs, minihack_recorder

    async def run_step(self, action: str, game_files: str, **kargs) -> Dict:
        """Run one step and log results."""
        obs, reward, done, info = self.envs[game_files].step(action)
        # print(obs['text'], reward,done)
        obs = obs['text']
        obs['action_took'] = action
        obs['scores'] = reward
        obs['dones'] = done
        
        recorder = self.recorders[game_files]
        await recorder.log_step(**obs)
        return  obs, recorder

    async def cal_reward(self, scores: float, game_files: str) -> MiniHackRecorder:
        """Store a simple binary reward and return recorder."""
        recorder = self.recorders[game_files]
        await recorder.set_reward(1.0 if scores == 1.0 else 0.0)
        return recorder

    async def get_prompt(self, game_files: str, long_term_context: str, short_term_context: str, memory_retrived: Dict = {}, **kargs):
        return get_minihack_prompt(
            env = self.envs[game_files], 
            task = game_files,
            long_term_context=long_term_context,
            short_term_context=short_term_context,
            memory_retrived=memory_retrived,
            **kargs)
        
def from_yaml(yaml_path: str):
    """Load record_db from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config
