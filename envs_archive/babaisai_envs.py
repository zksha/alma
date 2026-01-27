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

try:
    from eval_envs.prompts.env_prompt import get_babaisai_prompt
    import gym
    import baba
    from baba.world_object import name_mapping
    from baba import make
except Exception as e:
    pass

log = get_logger("main")

@dataclass
class BabaisaiRecorder(Basic_Recorder):
    """Recorder for a single environment interaction session."""

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Stores the initial observation. ",
            "type": "Dict[str, Any]",
            "example":{
                'goal': '- Identify the current rules, which are formed by text blocks in the format ”[Subject] IS [Property]” (e.g. ”BABA IS YOU”).\nYour goal is usually to reach an object defined as ”WIN”, but this can be changed.\nThe following are the possible actions you can take in the game, followed by a short description of each action:  idle: wait for one step, up: take one step up, right: take one step to the right, down: take one step down, left: take one step to the left.',
                'obs': 'Observation:  Active rules: baba is you  Objects on the map:  rule ‘is‘ 1 step to the left and 1 step up rule ‘win‘ 1 step up rule ‘key‘ 1 step to the left key 1 step to the right and 2 steps down ball 2 steps to the right and 3 steps down rule ‘baba‘ 2 step to the left and 4 steps down rule ‘is‘ 1 step to the left and 4 steps down rule ‘you‘ 4 steps down rule ‘ball‘ 2 steps to the right and 4 steps down.', 
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
                'action_took': ['left'], 
                'obs': 'Observation:  Active rules: baba is you  Objects on the map:  rule ‘is‘ 1 step to the left and 1 step up rule ‘win‘ 1 step up rule ‘key‘ 1 step to the left key 1 step to the right and 2 steps down ball 2 steps to the right and 3 steps down rule ‘baba‘ 2 step to the left and 4 steps down rule ‘is‘ 1 step to the left and 4 steps down rule ‘you‘ 4 steps down rule ‘ball‘ 2 steps to the right and 4 steps down.', 
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

class Babaisai_Env(Basic_Env):
    """Asynchronous babaisAI environment wrapper, returns a recorder at the end of episode."""

    def __init__(self,
        train_size, 
        train_eval: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "train",
        update_task: Literal["train", "eval_in_distribution", "eval_out_of_distribution"] = "eval_out_of_distribution",
        max_trails: int = 20,
        **kwargs
        ):   

        self.project_root = Path(__file__).resolve().parent.parent
        self.max_trails = max_trails
        config_path = self.project_root / 'eval_envs' / 'configs' / 'env_configs.yaml'
        self.babaisai_kwargs = from_yaml(config_path)
        self.task_list = list(baba.make("env/*").keys())
        # print(self.task_list,len(self.task_list))

        # register envs
        self.category_to_tasks = defaultdict(list)
        self.category_to_tasks['default'] = self.task_list

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
    
        env = make(task.rsplit("_runtime_", 1)[0], **self.babaisai_kwargs)
        env = BabaIsAIWrapper(env)

        babaisai_recorder = BabaisaiRecorder()

        obs = env.reset()
        obs['goal'] = '- Identify the current rules, which are formed by text blocks in the format ”[Subject] IS [Property]” (e.g. ”BABA IS YOU”).\nYour goal is usually to reach an object defined as ”WIN”, but this can be changed.\nThe following are the possible actions you can take in the game, followed by a short description of each action:  idle: wait for one step, up: take one step up, right: take one step to the right, down: take one step down, left: take one step to the left.'
        
        await babaisai_recorder.log_init(**obs)

        self.recorders[task] = babaisai_recorder
        self.envs[task] = env

        return obs, babaisai_recorder

    async def run_step(self, action: str, game_files: str, **kargs) -> Dict:
        """Run one step and log results."""
        obs, reward, done, info = self.envs[game_files].step(action)
        # print(obs['text'], reward,done)
        obs['action_took'] = action
        obs['scores'] = reward
        obs['dones'] = done
        
        recorder = self.recorders[game_files]
        await recorder.log_step(**obs)
        return  obs, recorder

    async def cal_reward(self, reward: float, game_files: str) -> BabaisaiRecorder:
        """Store a simple binary reward and return recorder."""
        recorder = self.recorders[game_files]
        await recorder.set_reward(1.0 if reward > 0 else 0.0)
        return recorder

    async def get_prompt(self, obs: str, memory_retrived: Dict = {}, **kargs):        
        prompt = get_babaisai_prompt(
            obs=obs,
            memory_retrived=memory_retrived,
            **kargs)
        return prompt
        
def from_yaml(yaml_path: str):
    """Load record_db from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config

class BabaIsAIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, add_ruleset=True, vlm=False):
        super().__init__(env)
        self.add_ruleset = add_ruleset
        self.BABAISAI_ACTION_SPACE = [a.name for a in baba.grid.BabaIsYouEnv.Actions]
        self.language_action_space = self.BABAISAI_ACTION_SPACE[:]
        self.progression = 0.0
        self.target_plan = None

    @property
    def default_action(self):
        return self.BABAISAI_ACTION_SPACE[0]

    def get_text_action(self, action):
        return self.language_action_space[action.value]

    def get_ruleset(self):
        """
        Retrieve and format the ruleset for the current environment.

        This method extracts rules from the environment's grid ruleset,
        formats them into human-readable strings, and returns them as a
        single string with each rule on a new line.
        """
        rules = []
        for rule in self.env.grid._ruleset["_rule_"]:
            # all objects start with f, eg `fwall`, `fkey`...
            # are objects that can be manipulated, `wall` is used to indicate end of map
            if "object" not in rule:  # BabaIsAI bug fix
                continue
            name = rule["object"].removeprefix("f")
            named_property = name_mapping[rule["property"]]
            rules.append(f"{name} is {named_property}")

        return "\n".join(rules)

    def get_text_observation(self, obs):
        """
        Generate a text-based observation of the environment.

        This method creates a textual description of the environment,
        including the relative positions of various objects with respect
        to the player's position (represented by 'baba').
        """

        def find_objects(objects):
            obj = []
            for j in range(0, self.env.height):
                for i in range(0, self.env.width):
                    cell = self.env.grid.get(i, j)
                    if cell is not None and cell.type in objects:
                        if cell.type == "rule_object":
                            name = f"rule `{cell.name}`"
                        elif cell.type == "rule_is":
                            name = f"rule `{name_mapping[cell.name]}`"
                        elif cell.type == "rule_property":
                            name = f"rule `{name_mapping[cell.property]}`"
                        else:
                            name = cell.type
                        obj.append(((i, j), name))
            return obj

        def calculate_offsets(reference_position, positions):
            reference_position = np.asanyarray(reference_position)
            positions = np.asanyarray(positions)

            relative_positions = []
            for pos in positions:
                relative_positions.append(pos - reference_position)

            return relative_positions

        def form_description(relative_positions):
            def steps(v):
                return "steps" if v > 1 else "step"

            descriptions = []
            for pos in relative_positions:
                (x, y), name = pos
                name = name.removeprefix("f")

                x_direction = ""
                if x > 0:
                    x_direction = f"{x} {steps(x)} to the right"
                elif x < 0:
                    x_direction = f"{-x} {steps(x)} to the left"

                y_direction = ""
                if y > 0:
                    y_direction = f"{y} {steps(y)} down"
                elif y < 0:
                    y_direction = f"{-y} {steps(y)} up"

                description = ""
                if x_direction:
                    description = f"{name} {x_direction}"

                if y_direction:
                    if x_direction:
                        description += f" and {y_direction}"
                    else:
                        description = f"{name} {y_direction}"

                descriptions.append(description)

            return "\n".join(descriptions)

        you = None
        for rule in self.env.grid._ruleset["_rule_"]:
            if "property" not in rule:  # BabaIsAI bug fix
                continue
            named_property = name_mapping[rule["property"]]
            if named_property == "you":
                you = rule["object"]

        # TODO: we need to handle multilpe me)
        my_position = find_objects([you])
        if len(my_position) == 0:
            # We should reset the environment, as baba cannot legally move anymore
            return self.reset(), True
        my_position = my_position[0]
        other_positions = find_objects(
            [
                "fball",
                "fwall",
                "fdoor",
                "fkey",
                "rule_object",
                "rule_is",
                "rule_property",
            ]
        )
        offsets = calculate_offsets(my_position[0], [p[0] for p in other_positions])
        relative_positions = [(tuple(offset), pos[1]) for offset, pos in zip(offsets, other_positions)]
        text_observation = form_description(relative_positions)

        return text_observation, False

    def textworld_process_obsv(self, textworld_obsv):
        ruleset = self.get_ruleset()
        text_observation, reset = self.get_text_observation(textworld_obsv)
        prompt = "" if not reset else "[...] IS YOU rule was broken, environment reset"
        if self.add_ruleset:
            prompt += f"Active rules:\n{ruleset}\n\n"
        prompt += f"Objects on the map:\n{text_observation}"

        obs = defaultdict(lambda: None)

        obs["obs"] = prompt

        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.target_plan = self.env.target_plan
        self.progression = 0.0

        return self.textworld_process_obsv(obs)

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, done, info = self.env.step(action_int)

        if done:
            self.progression = 1.0 if reward > 0 else 0.0

        return self.textworld_process_obsv(obs), reward, done, info

    def get_stats(self):
        return {"target_plan": self.target_plan, "progression": self.progression}
