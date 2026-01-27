from typing import List, Dict
import json

ALFWORLD_BASIC_PROMPT = f"""
You are an agent in a new house. You will receive descriptions of:
    1. Your current location.
    2. Objects/Receptacle around you.
    3. Actions you can perform.
    4. The task you need to complete.

### Task
Your goal is to continuously choose the most appropriate action in order to accomplish the given task efficiently and safely.

### Actions Types
You can perform the following types of actions:
- go to [receptacle]
- take [object] from [receptacle]
- put [object] in/on [receptacle]
- open [receptacle]
- close [receptacle]
- toggle [object] on/off [receptacle]
- clean [object] with [receptacle]
- heat [object] with [receptacle]
- cool [object] with [receptacle]

Here, [object] refers to any object you can interact with, and [receptacle] refers to any container or furniture that can hold or interact with objects. You will be provided with a detailed list of available objects and receptacles at each step.

### Guidelines
- Always carefully review the current state and choose the most suitable action.
- You may think creatively or combine steps to achieve the goal efficiently.
- Output **only** the chosen action exactly in the provided action format. Do not add explanations or commentary.
"""

def get_alfworld_prompt(obs: str, actions_list: List[str], memory_retrived: Dict = {}, **kargs):
    if memory_retrived:
        memory_prompt = f"""
    Here are past experiences and trajectories that might be helpful for your decision:
    {json.dumps(memory_retrived, indent = 2, ensure_ascii = False)}
    """
    else: 
        memory_prompt = ''
    
    system_prompt = ALFWORLD_BASIC_PROMPT + memory_prompt
    
    action_list = '\n'.join(actions_list)
    user_prompt = f"""
    {obs}
    Here are the actions you can choose from:
    <Actions>
    {action_list}
    </Actions>
    """
    return [{'role':'system', 'content': system_prompt}, {'role':'user', 'content':user_prompt}]