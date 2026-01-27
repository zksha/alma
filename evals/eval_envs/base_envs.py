from dataclasses import dataclass, field
from typing import Dict, Any
import asyncio

class Basic_Env:
    def set_task_env(self, task_config: Dict):
        pass

    def run_step(self, env, action: Dict):
        pass

    def cal_reward(self, reward: Any):
        pass

    def get_prompt(self, **kargs):
        pass

@dataclass
class Basic_Recorder:
    init: Dict[str, Any] = field(default_factory=dict)
    steps: list = field(default_factory=list)
    reward: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    memory_retrieved: Dict = field(default_factory=dict)

    async def log_init(self, **kargs):
        pass
    
    async def log_step(self, **kargs):
        pass

    async def set_reward(self, **kargs):
        pass

