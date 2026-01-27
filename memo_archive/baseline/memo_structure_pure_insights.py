from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
import asyncio
from typing import Dict, Optional, List
from utils.hire_agent import Agent
import re
import json

# prompt from <Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory>: https://github.com/suzgunmirac/dynamic-cheatsheet/blob/main/prompts/curator_prompt_for_dc_cumulative.txt
INSIGHTS_MODIFICATION_PROMPT = f"""# CHEATSHEET REFRENCE CURATOR

#### 1. Purpose and Goals
As the Cheatsheet Curator, you are tasked with creating a continuously evolving reference designed to help solve a wide variety of tasks, including algorithmic challenges, debugging, creative writing, and more. The cheatsheet's purpose is to consolidate verified solutions, reusable strategies, and critical insights into a single, well-structured resource.

- The cheatsheet should include quick, accurate, reliable, and practical solutions to a range of technical and creative challenges. 
- After seeing each input, you should improve the content of the cheatsheet, synthesizing lessons, insights, tricks, and errors learned from past problems and adapting to new challenges.

---

#### 2. Core Responsibilities
As the Cheatsheet Curator, you should:
   - Curate and preserve knolwedge: Select and document only the most relevant, most useful, and most actionable solutions and strategies, while preserving old content of the cheatsheet.
   - Maintain accuracy: Ensure that all entries in the cheatsheet are accurate, clear, and well-contextualized. 
   - Refine and update content: Continuously update and improve the content of the cheatsheet by incorporating new insights and solutions, removing repetitions or trivial information, and adding efficient solutions.
   - Ensure practicality and comprehensiveness: Provide critical and informative examples, as well as efficient code snippets and actionable guidelines. 

Before updating the cheatsheet, however, you should first assess the correctness of the provided solution and strategically incorporate code blocks, insights, and solutions into the new cheatsheet. Always aim to preserve and keep correct, useful, and illustrative solutions and strategies for future cheatsheets.

---

#### 3. Principles and Best Practices
1. Accuracy and Relevance:
   - Only include solutions and strategies that have been tested and proven effective.
   - Clearly state any assumptions, limitations, or dependencies (e.g., specific Python libraries or solution hacks).
   - For computational problems, encourage Python usage for more accurate calculations.

2. Iterative Refinement:
   - Continuously improve the cheatsheet by synthesizing both old and new solutions, refining explanations, and removing redundancies.
   - Rather than deleting old content and writing new content each time, consider ways to maintain table content and synthesize information from multiple solutions.
   - After solving a new problem, document any reusable codes, algorithms, strategies, edge cases, or optimization techniques. 

3. Clarity and Usability:
   - Write concise, actioanble, well-structured entries.
   - Focus on key insights or strategies that make solutions correct and effective.

4. Reusability:
   - Provide clear solutions, pseudocodes, and meta strategies that are easily adaptable to different contexts.
   - Avoid trivial content; focus on non-obvious, critical solution details and approaches.
   - Make sure to add as many examples as you can in the cheatsheet. 
   - Any useful, efficient, generalizable, and illustrative solutions to the previous problems should be included in the cheatsheet.

---

#### 4. Cheatsheet Structure
The cheatsheet can be divided into the following sections:

1. Solutions, Implementation Patterns, and Code Snippets:
   - Document reusable code snippets, algorithms, and solution templates.
   - Include descriptions, annotated examples, and potential pitfalls, albeit succinctly.

2. [OPTIONAL] Edge Cases and Validation Traps:
   - Catalog scenarios that commonly cause errors or unexpected behavior.
   - Provide checks, validations, or alternative approaches to handle them.

3. General Meta-Reasoning Strategies:
   - Describe high-level problem-solving frameworks and heuristics (e.g., use Python to solve heuristic problems; in bipartite graphs, max matching = min vertex cover, etc.)
   - Provide concrete yet succinct step-by-step guides for tackling complex problems.

4. Implement a Usage Counter
   - Each entry must include a usage count: Increase the count every time a strategy is successfully used in problem-solving.
   - Use the count to prioritize frequently used solutions over rarely applied ones.

---

#### 5. Formatting Guidelines
Use the following structure for each memory item:

```
<memory_item>
<description>
[Briefly describe the problem context, purpose, and key aspects of the solution.] (Refence: Q1, Q2, Q6, etc.)
</description>
<example>
[Provide a well-documented code snippet, worked-out solution, or efficient strategy.]
</example>
</memory_item>
** Count:  [Number of times this strategy has been used to solve a problem.]


<memory_item>
[...]
</memory_item>

[...]

<memory_item>
[...]
</memory_item>

```

- Tagging: Use references like `(Q14)` or `(Q22)` to link entries to their originating contexts.
- Grouping: Organize entries into logical sections and subsections.
- Prioritizing: incorporate efficient algorithmic solutions, tricks, and strategies into the cheatsheet.
- Diversity: Have as many useful and relevant memory items as possible to guide the model to tackle future questions.

N.B. Keep in mind that once the cheatsheet is updated, any previous content not directly included will be lost and cannot be retrieved. Therefore, make sure to explicitly copy any (or all) relevant information from the previous cheatsheet to the new cheatsheet!!!

---

#### 6. Cheatsheet Template
Use the following format for creating and updating the cheatsheet:

NEW CHEATSHEET:
```
<cheatsheet>

Version: [Version Number]

SOLUTIONS, IMPLEMENTATION PATTERNS, AND CODE SNIPPETS
<memory_item>
[...]
</memory_item>

<memory_item>
[...]
</memory_item>

GENERAL META-REASONING STRATEGIES
<memory_item>
[...]
</memory_item>

</cheatsheet>
```

N.B. Make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block. The cheatsheet can be as long as circa 2000-2500 words. Just output the NEW CHEATSHEET.
"""

class InsightsMemory(MemoStructure):
    def __init__(self):
        super().__init__()
        self.insights_db = {'insights':''}
        self.db_lock = asyncio.Lock()

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        return self.insights_db

    async def general_update(self, recorder: Basic_Recorder) -> None:
      reward = recorder.reward
      traj = f"""# INITIAL ENV
        {json.dumps(recorder.init, indent = 1, ensure_ascii = False)}
        # STEPWISE RECORD
        {json.dumps(recorder.steps, indent = 1, ensure_ascii = False)}
        """
      user_prompt = f"""
## PREVIOUS CHEATSHEET

{self.insights_db.get('insights')}

-----
-----

## CURRENT TRAJECTORY

{traj}

-----
-----

## REWARD TO CURRENT TRAJECTORY
{reward}"""
      def extract_cheatsheet(text: str):
         match = re.search(r"<cheatsheet>(.*?)</cheatsheet>", text, re.DOTALL)
         if match:
               return "<cheatsheet>" + match.group(1).strip() + "</cheatsheet>"
         return None

      agent = Agent(
         system_prompt = INSIGHTS_MODIFICATION_PROMPT,
         model = 'gpt-4o-mini'
      )
      max_retries = 3

        
      for attempt in range(max_retries):
         response = await agent.ask(user_input=user_prompt)

         parsed = extract_cheatsheet(response)
         async with self.db_lock:
            self.insights_db['insights'] = parsed
            return

      return