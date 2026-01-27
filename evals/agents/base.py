import asyncio
from curses import meta
from pickle import GLOBAL
from typing import List, Dict, Any, Optional
from typing_extensions import Literal
from collections import defaultdict
from logger import get_logger
log = get_logger("main")
from dataclasses import dataclass, field
import openai

@dataclass
class TokenTracker:
    def __init__(self):
        self.model_usage = defaultdict(lambda: {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0
        })
        
    async def update(self, model_name: str, usage: dict):
        """
        Safely update token counts with a lock to prevent race conditions.
        """
        entry = self.model_usage[model_name]

        def get(v, key):
            if isinstance(v, dict):
                return v.get(key, 0)
            return getattr(v, key, 0)

        entry["total_tokens"] += get(usage, "total_tokens")
        entry["prompt_tokens"] += get(usage, "prompt_tokens")
        entry["completion_tokens"] += get(usage, "completion_tokens")

        completion_details = get(usage, "completion_tokens_details")
        if completion_details:
            reasoning_tokens = get(completion_details, "reasoning_tokens")
            entry["reasoning_tokens"] += reasoning_tokens
    
    def summary(self):
        """Return a readable summary of token usage per model."""
        result = {}
        for model, stats in self.model_usage.items():
            result[model] = dict(stats)
        return result
    
    def print_summary(self):
        log.info("[blue]━━━━━━━━━━━━━━━ Token Usage Summary ━━━━━━━━━━━━━━━[/blue]")
        for model, stats in self.model_usage.items():
            log.info(f"[purple]Model: {model} | Prompt Tokens: {stats['prompt_tokens']} | Completion Tokens: {stats['completion_tokens']}[/purple]")

GLOBAL_TOKEN_TRACKER = None

def init_global_tracker():
    """Only called after all modules are imported"""
    global GLOBAL_TOKEN_TRACKER
    if GLOBAL_TOKEN_TRACKER is None:
        GLOBAL_TOKEN_TRACKER = TokenTracker()
    return GLOBAL_TOKEN_TRACKER
