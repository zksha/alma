import sys
from pathlib import Path
nle_path = Path("/opt/balrog/balrog/environments").resolve()
if str(nle_path) not in sys.path:
    sys.path.insert(0, str(nle_path))
from typing import List, Dict, Optional
import json


ACTIONS = {
    "idle": "wait for one step",
    "up": "take one step up",
    "right": "take one step to the right",
    "down": "take one step down",
    "left": "take one step to the left",
}


def get_babaisai_prompt(obs: str, memory_retrived: Dict = {}, **kargs):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())
    if memory_retrived:
        memory_prompt = f"""
    **Relevant Experience: **
    Here are past experiences and trajectories that might be helpful for your decision:
    {json.dumps(memory_retrived, indent = 2, ensure_ascii = False)}
    PLAY!
    """
    else: 
        memory_prompt = '\nPLAY!'
    instruction_prompt = f"""
Baba Is You is a puzzle game where you can manipulate the rules of each level. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.
- Only output the available action. No explaination is needed.
""".strip()
    instruction_prompt = instruction_prompt + memory_prompt
    user_prompt = "\nObservation:\n" + obs
    return [{'role':'system','content':instruction_prompt.strip()}, {'role':'user','content':user_prompt.strip()}]