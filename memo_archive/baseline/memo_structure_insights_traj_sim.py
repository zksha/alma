from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
import asyncio
from typing import Dict, Optional, List
from utils.hire_agent import Agent
import re
import json

# prompt from <ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory>
GET_SUCCESS_PROMPT = f"""You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the  generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
``` 
# Memory Item i 
## Title <the title of the memory item> 
## Description <one sentence summary of the memory item> 
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task> 
```
"""

GET_FAIL_PROMPT = f"""You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the  generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
``` 
# Memory Item i 
## Title <the title of the memory item> 
## Description <one sentence summary of the memory item> 
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task> 
```
"""

class InsightsMemory(MemoStructure):
   def __init__(self):
      super().__init__()
      self.insights_db = {}
      self.db_lock = asyncio.Lock()

   async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
      async def get_embedding(text: str) -> List[float]:
         from utils.hire_agent import Embedding
         embedder = Embedding()
         return await embedder.get_embedding(text)

      async def best_match_by_embedding(target: str, candidates: List[str]) -> Optional[str]:
         if not candidates:
               return None
         try:
               from utils.hire_agent import Embedding
               emb_cands = await Embedding().get_batch_embeddings(candidates)
               emb_target = await get_embedding(target)
               if not emb_target or not emb_cands: return None
               from utils.hire_agent import Embedding as EClass
               sims = await EClass.compute_one_to_group_similarity(emb_target, emb_cands)
               best_idx = max(range(len(candidates)), key=lambda i: sims[i])
               return candidates[best_idx] if sims[best_idx] > 0.3 else None  # loose cutoff
         except Exception:
               return None
      
      init_dict = recorder.init
      init_obs = ''
      for key, val in init_dict.items():
         init_obs += f'\n<{key}>{val}</{key}>'
      best_candidate = await best_match_by_embedding(init_obs, list(self.insights_db.keys()))
      if best_candidate:
         memo = {
            'retrieved_similar_tasks':self.insights_db[best_candidate]
         }
         return memo
      else: 
         return {}
         

   async def general_update(self, recorder: Basic_Recorder) -> None:
      init_dict = recorder.init
      init_obs = ''
      for key, val in init_dict.items():
         init_obs += f'\n<{key}>{val}</{key}>'

      traj = f"""# INITIAL ENV
      {json.dumps(recorder.init, indent = 1, ensure_ascii = False)}
      # STEPWISE RECORD
      {json.dumps(recorder.steps, indent = 1, ensure_ascii = False)}
      """
      user_prompt = f"""
      ## CURRENT TRAJECTORY
      {traj}
      ## REWARD TO CURRENT TRAJECTORY
      {recorder.reward}"""
      
      if recorder.reward == 1.0:
         agent = Agent(
               system_prompt = GET_SUCCESS_PROMPT,
               model = 'gpt-4o-mini'
         )
      else:
         agent = Agent(
               system_prompt = GET_FAIL_PROMPT,
               model = 'gpt-4o-mini'
         )
      max_retries = 3
      for attempt in range(max_retries):
         response = await agent.ask(user_input=user_prompt)
         async with self.db_lock:
            self.insights_db[init_obs] = response
            return

      return