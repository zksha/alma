from doctest import Example
import json
from typing import Any, Dict, List
from dataclasses import fields
from pathlib import Path
from evals.eval_envs.base_envs import Basic_Recorder

TASK_DESCRIPTION = {
        'alfworld': """The evaluation of downstream agent system is based on TextWorld: 
- The agent explores within a given space (rooms) and a predefined list of actions in order to accomplish a specified task goal. 
- Different data may involve **different rooms/spaces**, **different object placements**, or **entirely different task objectives**.
- The memory structure should strenthen following abilities of agent:
    - Spatial and State Reasoning: what are possible actions for each type of object?
    - Object-Centered Trajectory Learning: what did the agent do for previous same object management tasks or same location tasks?
""",
    'minihack':"""The evaluation of downstream agent system is based on MiniHack:
    - MiniHack is a collection of small game-like tasks where an agent sees a grid map of the environment, text and color representations, its own stats like health and position, and its inventory.  
    - Different games will have **different goals, maps and observations**. So goals and initial observations can change drastically across episodes, requiring **transferable knowledge**.
    - The memory structure should strenthen following abilities of agent:
        - Spatial and State Reasoning: Infer the environment from partial observations, remember explored areas, and navigate the map effectively.
        - Long-Horizon Goal Planning: Form multi-step strategies and consistently act toward the task objective despite sparse or delayed rewards.
        - Environmental Interaction and Risk Management: Correctly use actions and make safe, context-aware decisions around monsters, traps, and terrain.
    """,
    'textworld':"""The evaluation of downstream agent system is based on TextWorld:
    - The agent explores within a given space (rooms) and actions in order to accomplish a specified task goal. The agent may need to explore the space and get more progession. 
    - Done the task not neccessarily means the agent win the game. Whether wining or not depends on final reward(a progression percentage).
    - Different data may involve **different rooms/spaces**, **different object placements**, or **entirely different task objectives**.
    """,
    'babaisai':"""The evaluation of downstream agent system is based on BABAisAI:
    - In this gridworld game, agent interact with various objects and textual rule blocks to achieve specific goals.
    - Different data may involve **different rooms/spaces**, **different object placements**.
    - The agent goal is usually to reach an object defined as ”WIN”, but this can be changed by changing the rules.
    - The rules of the game can be manipulated and rearranged by the agent, creating a dynamic environment where agents must identify relevant objects and rules and then manipulate them to change or create new rules to succeed.
    - The memory structure should strenthen following abilities of agent:
        - ability to perceive all objects and text blocks in the environment, understand which rules are currently active or blocked, and evaluate how the environment state affects potential winning conditions.
        - ability to analyze and determine whether it is possible to win by modifying rule text blocks, and identify which rule combinations could achieve the goal.
        - ability to plan multi-step actions, including moving the character and pushing text blocks, to gradually construct or alter rules that lead to victory.
        - ability to flexibly adapt its strategy in response to dynamic rule and environment changes, handling potential obstacles or conflicts, and learn when modifying rules is more effective than simple movement.
        - ability to possess exploration and strategy transfer capabilities, discovering new ways to manipulate rules in unseen levels and applying prior experience to recognize when rule changes are advantageous.
    """
    }

# read from tools and cheetsheet
CHROMA_CHEETSHEET = """## Initialize Chroma DB

Use db = Chroma(embedding_function=Embedding()) to create the database. DO NOT use persist_dir.

### Add Memory: Adds new text entries to the database and returns their unique IDs. 

db.add_texts(
    texts: List[str],
    metadatas: Optional[Union[str, int, float, bool, None]] = None,
    ids: Optional[List[str]] = None
) -> List[str]

- metadatas must be **flat list**: each value must be a single primitive type (str, int, float, bool, or None).
- You cannot pass lists, nested dicts, or other complex objects.
- If you need to store structured data, serialize it to a JSON string:

### Retrieve Memory

db.similarity_search(
    query: str,
    k: int = 4
) -> List[Document]

return List[Document]: [
  Document(
    page_content="the agent found a key",
    metadata={"type": "item"}
  )
]

### Get by ID
db.get(
    ids: Optional[List[str]] = None
) -> Dict[str, List]

### Delete Memory
db.delete(
    ids: Optional[List[str]] = None
) -> None
    """

NXGRAPH_CHEETSHEET = """
NETWORKX GRAPH CHEATSHEET

Context: 
    import networkx as nx
    G = nx.Graph()

1. NODE OPERATIONS
- G.add_node(node, **attrs): Add a single node with optional attributes.
- G.add_nodes_from([n1, n2], **common_attrs): Add multiple nodes at once (shared attributes apply to all).
- G.remove_node(node): Remove a node and all edges connected to it.
- G.remove_nodes_from([n1, n2]): Remove multiple nodes.
- node in G: Check if a node exists.
- G.nodes: Get all nodes (NodeView).
- G.nodes[node]: Access node attributes as a dict.
- nx.set_node_attributes(G, {node: {"attr": value}}): Set attributes for nodes.

2. EDGE OPERATIONS
- G.add_edge(u, v, **attrs): Add an edge between two nodes.
- G.add_edges_from([(u, v), (x, y)], **attrs): Add multiple edges at once (shared attributes apply to all).
- G.remove_edge(u, v): Remove a single edge.
- G.remove_edges_from([(u, v), (x, y)]): Remove multiple edges.
- G.has_edge(u, v): Check if an edge exists.
- G.edges: Get all edges (EdgeView).
- G.edges[(u, v)]: Access edge attributes as a dict.
- nx.set_edge_attributes(G, {(u, v): {"weight": 1.0}}): Set attributes for edges.

3. TRAVERSAL / NEIGHBORHOOD
- G.neighbors(node): Get neighbors of a node.
- G.adj[node]: Get dict of neighbors with edge data.
- nx.shortest_path(G, source, target): Find one shortest path between nodes.
- nx.shortest_path_length(G, source, target): Get shortest path length.
- nx.all_simple_paths(G, source, target, cutoff): Generate all simple paths up to a cutoff length.
- nx.connected_components(G): Get connected components as node sets.
- G.subgraph([n1, n2, n3]): Extract a subgraph induced by given nodes.

4. ANALYSIS / CENTRALITY
- G.degree(node): Get degree (number of edges) for a single node.
- G.degree(): Get degree for all nodes (DegreeView).
- nx.degree_centrality(G): Compute degree centrality (dict of node -> score).
- nx.betweenness_centrality(G): Compute betweenness centrality.
- nx.pagerank(G): Compute PageRank scores for nodes.
- nx.clustering(G): Compute local clustering coefficient.
- nx.is_connected(G): Check if graph is connected.
- nx.number_connected_components(G): Count connected components.

5. UTILITIES
- G.copy(): Make a copy of the graph.
- G.clear(): Remove all nodes and edges.
- nx.to_dict_of_dicts(G): Convert graph to adjacency dict.
- nx.to_numpy_array(G): Get adjacency matrix as a NumPy array.
"""

TOOL_CHEETSHEET = """
TOOLS AVAILABLE:

1. Hire Agent
Class: Agent

- Purpose: Asynchronous wrapper around OpenAI Chat API, allows system/user prompts and optional JSON schema validation. Higher insights: this could be used to summarise information or gain new insights from them.
- Initialization: 
    from utils.hire_agent import Agent 
    agent = Agent(model: str, system_prompt: str, output_schema: Optional[Dict] = None)
- Key Methods:
    - agent.get_agent_config() -> Dict: Returns agent configuration and chat history.
    - await agent.ask(user_input: str, with_history: bool = False) -> Any:
        Sends a user message asynchronously and returns the model's response.
        If an output_schema is provided, response will be validated against it and returned as a JSON object.
- Usage Example:
    response = await agent.ask("Hello, how are you?")

-IMPORTANT: 
    - output_schema must be a **json schema**. Format example:
{
    "location": {
        "type": "string",
        "description": "The location to get the weather for"
    },
    "unit": {
        "type": ["string", "null"],
        "description": "The unit to return the temperature in",
        "enum": ["F", "C"]
    }
}

    - model could be "gpt-4.1", "gpt-4o-mini", based on whether you need reasoning ability(if reasoning ability is needed, gpt-4o-mini could be a better choice).

2. Embedding
Class: Embedding 

- Purpose: Async embedding manager for computing single or batch embeddings, with optional similarity calculation. Higher insights: this could be used to find connections between texts.
- Initialization:
    from utils.hire_agent import Embedding
    embedder = Embedding(model: str = "text-embedding-3-small", retries: int  = 3, retry_delay: float = 1.0)
- Key Methods:
    - await embedder.get_embedding(text: str) -> List[float]:
        Computes embedding for a single text string asynchronously.
    - await embedder.get_batch_embeddings(texts: List[str]) -> List[List[float]]:
        Computes embeddings for multiple texts asynchronously.
    - await Embedding.compute_similarity(emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        Computes similarity between two embeddings asynchronously.
    - await Embedding.compute_one_to_group_similarity(emb: List[float], group_emb: List[List[float]], metric: str = "cosine") -> List[float]:
        Computes similarity between one embedding and a group of embeddings asynchronously.

- Notes:
    * All embedding calls automatically update a global token tracker.
    * Similarity functions support cosine similarity and run in parallel for efficiency.
"""

def build_analysis_prompt(memo_info, task_type):
    MEMO_ANALYSIS_OUTPUT_FORMAT = {
        "learned_from_suggestion_example": {
        "type": "string",
        "description": "Findings derived from the provided suggestion_example and improve_score. Bullet list of concrete factors (patterns) that made the suggestion succeed or fail. And principles to adopt when making future suggestions."
    },
    "trajectory_score_assessment": {
        "type": "array",
        "description": "analysis each retrieved module information based on current trajectories sampled and the benchmark scores.",
        "items": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["Useful", "Potentially Useful", "Irrelevant", "Empty/BadFormat"],
                    "description": "Categorization of the memory item's relevance based on whether the retrieved content actually helps the agent."
                },
                "how_it_can_help": {
                    "type": "string",
                    "description": "If Useful/Potentially Useful: short note how it could guide actions (subgoal, trap, object use...). If Irrelevant/Empty: reason (e.g., wrong keying, over-specific, missing summary, formatting)."
                }
            },
            "required": ["label", "how_it_can_help"]
        }
    },
    "content_quality_issues": {
        "type": "string",
        "description": "Detected content-level problems (duplicates, empty entries, serialization issues...). Why those harms retrieval or downstream planning.",
    },
    "structure_and_coherence": {
        "type": "string",
        "description": "Analysis of layer interactions, keying, and task-awareness. Which parts generalize, which are overfitted.",
    },
    "suggested_changes": {
        "type": "array",
        "description": "Based on all your analysis above, provide concrete change that can be applied on provided current memory structure code.",
        "items": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "How urgent/impactful this is."},
                "what": {"type": "string", "description": "Precise description of what to change (code/pipeline/config)."},
                "why": {"type": "string", "description": "Link to observations/principles: why this addresses the problem."}
            },
            "required": ["priority", "what", "why"]
        }
    }
}

    example = memo_info.get('improve_example',{})
    example = f"""<Suggestion Example>
    Here is a previous suggestion, along with the code it looked at. 4. 
    The example include a memory structure and it's modification attempt, annotated with an improvement score (positive = improved, negative = degraded). 
    Infer the underlying patterns that differentiate effective modifications from harmful ones, and apply this reasoning to suggest an improved modification for the current memory structure.
    {json.dumps(example, indent=1, ensure_ascii=False)}
    </Suggestion Example>
    """ if example else ''
    # print(example)

    system_prompt = f"""You are a **Senior Agent Construction Engineer** responsible for provide suggestions for a memory structure written by a entry level engineer, to make the memory structure better for downstream agent to finish tasks.
### Memo Information Overview
1. **source_code**
    - Each layer(which inherits `Sub_memo_layer`) contains:
      - **Retrieve**: fetches relevant memory elements from the database.
      - **Update**: writes or modifies entries in the database.
    - Final MemoStructure(which inherits `MemoStructure`) contains:
        - **general_retrieve**: general retrieve method, that contain the order and input-output usage of retrieve function of each layer.
        - **general_update**: general update method, that contain the order and input-output usage of update function of each layer.
    - code usage: Your memory structure will be used in the agent workflow:
        - `general_retrieve(recorder)`: used **before** start executing the task, to retreive task relevant information. Your output json will be directly send to agent, so please make sure your output is well organized, include all useful information and avoid redundency, in a agent understandable way.
        - `general_update(recorder)`: used **after** task is finished, to update the trajectory, reward, or other information.
2. **examples**
    - **examples**: some trajectries, including the memory retrieved(memory retrived should only happened before starting the task), and the steps of actions took by agent.
3. **benchmark_eval_score**
    -  performance(success rate) of current memory stucture + general agent system. Need to use the score to analyze the performance and bottleneck of current memory structure.

### Your Task:
You will analyze past suggestion examples(including past source code, suggestions, and the improve score it led to) and the current retrieved trajectories and memory source code, then produce concrete, prioritized suggestions to improve the memory structure.
Follow the numbered procedure below and produce the requested structured outputs.
Step 1 — Learn from past suggestions & the improve score
    1. Look at the provided improve_score (positive → improvement, negative → degradation) and the single suggestion_example that produced that score.
    2. Explain why that suggestion led to improvement or degradation:
        - What pattern in the change made it succeed or fail?
        - Which behaviors, assumptions, or shortcuts in that suggestion were helpful? Which were harmful?
        - From these concrete cases, extract 2–5 general principles to adopt and 2–5 pitfalls to avoid when creating future suggestions.

Step 2 — Inspect sampled trajectories and benchmark performance and decide which memories are useful
    1. Review the current trajectory_examples (they include episodes with varying rewards).
    2. For each retrieved memory item (or memory group) returned for the trajectory, label it as one of:
        - Useful & Relevant — clearly applies to the current situation and can guide action;
        - Potentially Useful — has value but needs reformatting, summarization, or indexing to be helpful;
        - Irrelevant / Confusing — not related to this trajectory or misleading;
        - Empty / Badly Formatted — blank, placeholder, or not parseable. 
    3. For each memory you mark Useful/Potentially Useful, say how it would help (e.g., provides a repeated subgoal, highlights a trap, identifies key object interactions).
    4. For Irrelevant/Empty items, explain why they failed retrieval combine with the memory source code (e.g., wrong keying, over-specific content, missing summarization).

Step 3 — Inspect memory source and produce concrete suggestions
    1. Review the memory source code (retrieval keys, indexing, storage format, layers). Using Step 1 principles and Step 2 labels, propose specific changes to the memory system that address the observed issues.
    2. Combined the memory source code with your analysis in step 2, giving suggestions. For each suggested change, include:
        - What to change (code-level or pipeline change, e.g., add summarization layer, change indexing key, normalize objects to noun-phrases).
        - Why it will help (link back to a principle or a concrete failing you observed).
    3. Prioritize suggestions: label them High / Medium / Low priority and give an implementation order.
    4. Link Analysis to Benchmark Performance
        - Use benchmark_eval_score to identify which structural weaknesses correlate with poor performance.

Extra checks (quality & coherence)
    1. Flag obvious content issues: duplicates, empty entries, raw dumps, mis-typed fields, or numeric types that break JSON serialization.
    2. Check layer interaction: do layers pass structured outputs to each other, or only dump free-form text?
    3. If retrieval returns empty lists or dicts, emphasize structural fixes (keying, ensure type consistency, avoid over-relying on try/except fallbacks).

Goal: Combine reflection on past improvement signals with current system diagnosis to produce actionable, high-level suggestions that strengthen memory structure quality.

### Benchmark Information:
{TASK_DESCRIPTION[task_type]}

### Required Output:
Return a JSON object followed below json schema:
{json.dumps(MEMO_ANALYSIS_OUTPUT_FORMAT, indent = 2, ensure_ascii = False)}
"""
    # print(memo_info["source_code"][:20])
    # print(memo_info.get('benchmark_eval_score').get('benchmark_overall_eval_score'))
    user_prompt = f"""{example}
<CURRENT SOURCE CODE>
{memo_info["source_code"]}
</CURRENT SOURCE CODE>
<CURRENT TRAJECTORY EXAMPLE>
{json.dumps(memo_info["examples"], indent=2, ensure_ascii=False)}
</CURRENT TRAJECTORY EXAMPLE>
<CURRENT BENCHMARK SCORE>
{memo_info.get('benchmark_eval_score').get('benchmark_overall_eval_score')}
</CURRENT BENCHMARK SCORE>
"""
    return system_prompt,  user_prompt, MEMO_ANALYSIS_OUTPUT_FORMAT

def get_metadata_dict(instance) -> dict:
    meta_dict = {}
    for f in fields(instance):
        meta_dict[f.name] = dict(f.metadata)
    return meta_dict

def build_generate_new_code_prompt(memo_info: Dict[str,Any], analysis_result: Dict[str,Any], recorder: Basic_Recorder, task_type: str):
    base_dir = Path(__file__).parent.parent / "evals" / "agents"
    file_path = base_dir / "memo_structure.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find memo_structure.py at {file_path.resolve()}")
    basic_classes = file_path.read_text(encoding="utf-8")
    # construct interaction block
    interaction_recorder_info = get_metadata_dict(recorder)
    if analysis_result:
        suggestion = {}
        suggestion['trajectory_score_assessment'] = analysis_result['trajectory_score_assessment']
        suggestion['suggested_changes'] = analysis_result['suggested_changes']
    interaction_prompt = f"""
    Your `general_retrieve` and `general_update` will take `Basic_Recorder` as input, which has following attributes:
    {json.dumps(interaction_recorder_info, indent=2, ensure_ascii=False)}
    - For `general_retrieve`, only leverage `.init` attribute.
    - For `general_update`, leverage `.init`, `.steps`, `reward` attribute.
    - Each element in the above dict is a attribute name as key, and description, type, and a example for the exact possible value the attribute could have.
    - please note that all provided current trajectory can already been seen by down stream agents(in history), your memory structure should focus on provide extra advice and reference for agents.
    """



    system_prompt = f"""You are a senior AI software engineer. Your task is to build an agent memory system composed of multiple specialized memory layers and a coordinating memory structure. The agent will be used in {task_type}. Yout memory structure aims to help down stream agent to have more relevant advice, experience to help it further finish current task.
    
{TASK_DESCRIPTION[task_type]}

You are given the following two base classes:
<BACKBONE_CODE>
{basic_classes}
</BACKBONE_CODE>
Inherit these two base classes and recorder by writing:
```python
from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from utils.hire_agent import Agent, Embedding
from langchain_chroma import Chroma
```

<CODE_INPUT>
{interaction_prompt}
</CODE_INPUT>

<CODE_USAGE>
Your memory structure will be used in the agent workflow:
    - `general_retrieve(recorder)`: used **before** start executing the task, to retreive task relevant information. Your output json will be directly send to agent, so please make sure your output is well organized, include all useful information and avoid redundency, in a agent understandable way.
    - `general_update(recorder)`: used **after** task is finished, to update the trajectory, reward, or other information.
<CODE_USAGE>

Here is the basic tools provided:
<GRAPH_DATABASE_INTERACTION>
{NXGRAPH_CHEETSHEET}
</GRAPH_DATABASE_INTERACTION>

<CHROMA_DATABASE_INTERACTION>
{CHROMA_CHEETSHEET}
</CHROMA_DATABASE_INTERACTION>

<OTHER_TOOLS>
{TOOL_CHEETSHEET}
</OTHER_TOOLS>

### Your Task:
Modify the code above so that it fully satisfies the following design goals:

1. **Multiple Memory Layers:**   
   - Create multiple subclasses of `Sub_memo_layer`.
   - Each layer must have a **clear responsibility** and you might need to maintain its own database(based on nx.graph or chroma).  
   - Think carefully about **what type of data** belongs in each layer's database.

2. **General Retrieve/Update Orchestration:**  
   - Create a subclass of `MemoStructure` that orchestrates all layers.  
   - `general_retrieve()` should intelligently chain the results:  
     - Output from layer 1 can become input to layer 2.  
     - You should design a reasonable order of retrieval.  
   - `general_update()` should propagate updates to relevant layers in a sensible order.  
   - You are free to reorder calls or preprocess inputs to achieve coherent memory behavior.

3. **Out-of-the-Box Reasoning:**  
   - Do not just mechanically call each layer one by one — think about the **semantic flow of information**.  
   - Consider cases like:  
     - what type of memory layer can be used according to the analysis result or task description, with the aim to better assist the agent to finish it's task? 
     - what order and input output should be best suitable for the analysis result or task description? 
     - Think about high-level stategy: what can be a good memory structure, and has good ability to transfer to other area? 
   - Make sure each layer plays a meaningful role in the system.
   - Directly perform simple plans or content writing based on if/else patterns should be avoided, since this will hurt the transfer ability.
   - Keep the retrieved memory clean and useful, aviod cutting off meaningful texts, repeat same patterns in the retrieved memory.

4. **Integration with Utilities:**  
   - Feel free to use any provided utility functions (e.g., similarity calculation, interaction with databases, hire new agent) if relevant. The tools available will be listed in `TOOLS` section.
   - You can also create your own tools if neccessary, think out of the box.

5. **Code Quality:**  
   - Output clean, runnable Python code following PEP8.  
   - Ensure `general_retrieve()` and `general_update()` accept `Basic_Recorder` and orchestrate the pipeline end-to-end.  
   - Initialize all layers in `MemoStructure.__init__`.
   - Do not overuse defensive programming; raise appropriate exceptions when unexpected conditions occur to facilitate debugging.

6. **Coherent Policy Logic:**
    - Avoid placeholders like pass or # TODO.
    - Avoid hard-coded if/else branches or enumerated case handling; instead, express the logic through modular policy functions, scoring mechanisms, or composable decision rules.
    - Instead of enumerating case-specific rules, express generalizable principles that could apply across different families or new unseen tasks.
    - The logic should be adaptable and compositional, not dependent on predefined constants or string names.
    - Use abstractions instead of specific family identifiers.

The goal is to ensure the memory policy behaves consistently across tasks and supports generalization, not to hard-code specific task behaviors.

### Important:
- Think creatively about data flow — outputs of one layer can feed into the next.
- Each layer's functionality and stored data should be clearly designed.
- Provide **only the final rewritten code**, no explanations.
"""
    
    if memo_info.get('source_code',''):
        # print(memo_info.get('benchmark_eval_score').get('benchmark_overall_eval_score'))
        user_prompt = f"""
        Here is the current code that you must edit:
        <CURRENT_CODE>
        {memo_info.get('source_code','')}
        </CURRENT_CODE>

        Here is the score of current code:
        <REWARD>
        {memo_info.get('benchmark_eval_score').get('benchmark_overall_eval_score')}
        </REWARD>

        Here is the analysis result (suggestions):
        <ANALYSIS_RESULT>
        {json.dumps(suggestion, ensure_ascii=False, indent=2)}
        </ANALYSIS_RESULT>
            """ 
    else:
        user_prompt = f"""Please generated new code based on your understanding about the task and requirements. """
    # print(system_prompt)
    return system_prompt, user_prompt

def build_reflection_prompt(code_str: str, recorder: Basic_Recorder, error_msg: str): 
    
    base_dir = Path(__file__).parent.parent / "evals/agents" 
    file_path = base_dir / "memo_structure.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find memo_structure.py at {file_path.resolve()}")
    basic_classes = file_path.read_text(encoding="utf-8")
    # construct interaction block
    interaction_recorder_info = get_metadata_dict(recorder)
    interaction_prompt = f"""
    Your `general_retrieve`, `general_update` will take `Basic_Recorder` as input, which has following attributes:
    {json.dumps(interaction_recorder_info, indent=2, ensure_ascii=False)}
    - For `general_retrieve`, only leverage `.init` attribute.
    - For `general_update`, you can leverage all listed attribute.
    """
    
    system_prompt = f"""You are a senior AI software engineer and code repair expert.
Your role is to carefully analyze the provided code and the error information, identify potential errors or design flaws, and directly rewrite or edit the code to fix those issues — while keeping the main design goals and intentions exactly the same.

You are given the following context and base classes:
<BACKBONE_CODE>
{basic_classes}
</BACKBONE_CODE>

<CODE_INPUT>
{interaction_prompt}
</CODE_INPUT>

<CODE_USAGE>
Your memory structure will be used in the agent workflow:
    - `general_retrieve(recorder)`: used **before** start executing the task, to retreive task relevant information. Your output json will be directly send to agent, so please make sure your output is well organized, include all useful information and avoid redundency, in a agent understandable way.
    - `general_update(recorder)`: used **after** task is finished, to update the trajectory, reward, or other information.
<CODE_USAGE>

### Your Task:
Carefully inspect the code, and error information, detect the root cause of the errors, structural issues, or missing implementations, and **only fix the root cause code**. Here are some cheetsheet that could be useful:

<GRAPH_DATABASE_INTERACTION>
{NXGRAPH_CHEETSHEET}
</GRAPH_DATABASE_INTERACTION>

<CHROMA_DATABASE_INTERACTION>
{CHROMA_CHEETSHEET}
</CHROMA_DATABASE_INTERACTION>

<OTHER_TOOLS>
{TOOL_CHEETSHEET}
</OTHER_TOOLS>

### Output:
Return **only the final corrected Python code**, no explanations or commentary.
"""

    user_prompt = f"""
    Here's the code with potential error:
    <CODE_FOR_MODIFY>
    ```python
    {code_str}
    ```
    </CODE_FOR_MODIFY>
    And here's corresponding error message:
    {error_msg}
    Find the root cause first and then modify only the corresponding code to avoid the error.
    """
    return system_prompt, user_prompt