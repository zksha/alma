from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
try:
    from memo_test.prompt.gmemory_prompt import GMemoryPrompts
except Exception as e:
    print(e)
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from utils.hire_agent import Embedding
from utils.hire_agent import Agent
import numpy as np
from finch import FINCH
import asyncio
import copy
import pickle
import random
import re
import os
from dataclasses import dataclass, field, asdict, replace
from collections import defaultdict
from typing import Any, Optional, Iterator, Union, Any, Iterable, Dict
import networkx as nx
import json
from networkx.readwrite import json_graph
import math
import yaml

import os
import uuid
import hashlib

def random_sha(n=8):
    return hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:n]

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_name: str) -> Union[list, dict]:

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, separators=(",", ": "))

def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []
    
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

@dataclass
class StorageNameSpace:
    """
    StorageNameSpace represents a namespace for storage-related tasks,
    such as indexing and querying.

    Attributes:
        namespace (str): The identifier for this storage namespace.
        global_config (dict): A dictionary containing global configuration
                              settings for the namespace.
    """
    namespace: str
    global_config: dict

    def _index_done(self):
        pass

    def _query_done(self):
        pass


@dataclass
class AgentMessage:
    """
    AgentMessage represents a structured message exchanged between agents,
    including optional instructions and metadata.

    Attributes:
        agent_name (Optional[str]): The name of the agent sending or receiving the message.
        system_instruction (Optional[str]): Optional system-level instruction guiding the agent's behavior.
        user_instruction (Optional[str]): Optional user-level instruction or query.
        message (Optional[str]): The core message content (response or statement).
        extra_fields (dict[str, Any]): A dictionary to hold additional custom fields or metadata.
    """
    agent_name: Optional[str] = None
    system_instruction: Optional[str] = None
    user_instruction: Optional[str]  = None
    message: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None) 

@dataclass
class StateChain:
    """
    Manages a chain of directed graph states representing the evolution of agent messages and their relationships.

    Each state is a NetworkX DiGraph, where nodes represent agent messages and edges represent connections (e.g., spatial edges)
    between agents. The class supports adding messages, moving to new states, and serializing/deserializing the chain.

    Attributes:
        chain_of_states (list[nx.DiGraph]): Internal list storing the sequence of graph states.
    """
    def __post_init__(self):
        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states = [initial_state]

    def __iter__(self) -> Iterator[nx.DiGraph]:
        return iter(self.chain_of_states)
    
    def __len__(self) -> int:
        return len(self.chain_of_states)

    def add_message(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
        
        current_state: nx.DiGraph = self._get_current_state()

        agent_message_dict: dict = asdict(agent_message)
        node_id = self._generate_node_id(agent_message.agent_name)
        current_state.add_node(node_id, **agent_message_dict)
        
        for up_node_id in upstream_agent_ids:
            if not current_state.has_node(up_node_id):
                raise ValueError("Upstream node does not exist.")
            current_state.add_edge(up_node_id, node_id, edge_type='spatial')
        return node_id

    def move_state(self, action: str, observation: str, **args) -> None:

        current_state: nx.DiGraph = self._get_current_state()
        current_state.graph.update({"action": action, "observation": observation, **args})

        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states.append(initial_state)
    
    def get_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError('Index out of range.')
        return self.chain_of_states[idx]

    def pop_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError('Index out of range.')
        return self.chain_of_states.pop(idx)
    
    @property
    def chain_of_states(self) -> list[nx.Graph]:
        return self._chain_of_states[:-1]
    
    def _generate_node_id(self, agent_name: str) -> str:
        current_state: nx.DiGraph = self._get_current_state()
        name_counter = current_state.graph["name_counter"]

        if agent_name not in name_counter:
            name_counter[agent_name] = 0
        else:
            name_counter[agent_name] += 1
        return f"{agent_name}-{name_counter[agent_name]}"

    def _get_current_state(self) -> nx.DiGraph:
        return self._chain_of_states[-1]
    
    @staticmethod
    def to_str(state_chain: "StateChain") -> str:
        return json.dumps([json_graph.node_link_data(state) for state in state_chain])

    @staticmethod
    def from_str(state_chain_str: str) -> "StateChain":
        state_chain = StateChain()
        state_chain._chain_of_states = [json_graph.node_link_graph(state_data) for state_data in json.loads(state_chain_str)]
        return state_chain


@dataclass
class MASMessage:
    """
    Represents a multi-agent system (MAS) message that encapsulates the main task, 
    its description, the trajectory of task actions and observations, and the 
    associated chain of states tracking the message evolution.

    Attributes:
        task_main (str): The main task or objective description.
        task_description (Optional[str]): Additional details or description of the task.
        task_trajectory (Optional[str]): A textual log of the sequence of actions and observations. Defaults to a prompt format.
        label (Optional[bool]): An optional label or flag associated with the task.
        chain_of_states (StateChain): A StateChain instance tracking the evolution of states/messages in the task.
        extra_fields (dict[str, Any]): A dictionary for storing additional arbitrary fields related to the message.
    """
    task_main: str
    task_description: Optional[str] = None
    task_trajectory: Optional[str] = '\n\n>'
    label: Optional[bool] = None
    chain_of_states: StateChain = field(default_factory=StateChain, repr=False)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    
    def add_message_to_current_state(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
        return self.chain_of_states.add_message(agent_message, upstream_agent_ids)
    
    def move_state(self, action: str, observation: str, **args) -> None:
        self.task_trajectory += f'{action}\n{observation}\n>'
        self.chain_of_states.move_state(action, observation, **args)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None)
    
    @staticmethod
    def to_dict(mas_message: "MASMessage") -> dict[str, str]:
        return {
            "task_main": mas_message.task_main,
            "task_description": mas_message.task_description,
            "task_trajectory": mas_message.task_trajectory,
            "label": mas_message.label,
            "extra_fields": json.dumps(mas_message.extra_fields),
            "state_chain": StateChain.to_str(mas_message.chain_of_states)
        }
    
    @staticmethod
    def from_dict(message_dict: dict) -> "MASMessage":
        return MASMessage(
            task_main=message_dict.get("task_main"),
            task_description=message_dict.get("task_description"),
            task_trajectory=message_dict.get("task_trajectory"),
            label=message_dict.get("label"),
            extra_fields=json.loads(message_dict.get("extra_fields", "{}")),
            chain_of_states=StateChain.from_str(message_dict.get('state_chain'))
        )

@dataclass
class GMemory(MemoStructure):
    """
    G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
    A three-tier hierarchical graph structure compo sed of the Insight Graph, Query Graph, and Interaction Graph.

    1. Interaction Graph - Trajectory Condensation: During the task-solving process, the multi-agent system (MAS) generates a chain of states, where each state represents a step in the process of arriving at the final answer. Behind each state is a corresponding message graph.
       Each task corresponds to a chain of states, which connects the middle and bottom layers of the multi-layer graph.
    2. Query Graph - Based on the current task, the system retrieves previously successful records. A k-hop approach is used to expand the search scope within the query graph.
    3. Insight Graph - Insights Retrieval: Relevant insights are retrieved based on the current task to assist in decision-making.
    """
    def __init__(self, **global_config):
        super().__init__()

        # Resolve persist directory
        self.persist_dir = global_config.get('persist_dir', None)
        if not self.persist_dir:
            self.persist_dir = os.path.join(os.path.dirname(os.getcwd()), "gmemory") #os.path.dirname(os.getcwd())
            rand = random_sha() 
            self.persist_dir = os.path.join(self.persist_dir, rand)
            os.makedirs(self.persist_dir, exist_ok=True)

        self._hop: int = global_config.get('hop', 1)
        self._start_insights_threshold: int = global_config.get('start_insights_threshold', 5)
        self._rounds_per_insights: int = global_config.get('rounds_per_insights', 5) 
        self._insights_point_num: int = global_config.get('insights_point_num', 5)
        self.embedding = Embedding()
        self.task_type: int = global_config.get('task_type', 'alfworld')

        self.main_memory = Chroma(
            embedding_function=Embedding(),
            persist_directory=self.persist_dir
        )

        self.task_layer = TaskLayer(
            working_dir=self.persist_dir,
            namespace='task_layer', 
            task_storage=self.main_memory
        )

        self.insights_layer = InsightsManager(
            working_dir=self.persist_dir, 
            namespace='insights', 
            task_storage=self.main_memory,
            task_layer=self.task_layer
        )

        self.insights_cache: list[str] = []

        print(self._get_hyperparams_dict())

    def init_task_context( 
        self, 
        task_main: str,    
        task_description: str = None,
    ) -> MASMessage:

        self.current_task_context = MASMessage(
            task_main=task_main,
            task_description=task_description
        )
        return self.current_task_context
    
    def add_agent_node(
        self, 
        agent_message: AgentMessage,
        upstream_agent_ids: list[str]
    ) -> str:
        node_id: str = self.current_task_context.add_message_to_current_state(agent_message, upstream_agent_ids)
        return node_id
    
    def move_memory_state(self, action: str, observation: str, **kargs) -> None:
        self.current_task_context.move_state(action, observation, **kargs)
    
    async def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        if self.current_task_context == None:
            raise RuntimeError('The current inside-trial memory is empty.')
        
        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f'\n- Environment feedback\n{feedback}\n'
        await self.add_memory(self.current_task_context)

        return self.current_task_context

    def summarize(self, **kargs) -> str:
        return self.current_task_context.task_description + self.current_task_context.task_trajectory
    
    
    def _get_hyperparams_dict(self) -> dict:
        return {
            'hop': self._hop,
            'start_insights_threshold': self._start_insights_threshold,
            'rounds_per_insights': self._rounds_per_insights,
            'insights_point_num': self._insights_point_num,
            'working_dir': self.persist_dir
        }


    async def add_memory(self, mas_message: MASMessage) -> None:
        """
        Add the mas_message corresponding to a completed task into memory:
        Step 1: Sparsification - remove incorrect steps
        Step 2: Add the sparsified trajectories to memory
        Step 3: If the number of steps in memory reaches a certain threshold, perform fine-tuning on the insights in memory

        Args:
            mas_message (MASMessage): The MAS message corresponding to a completed task

        Raises:
            ValueError: mas_message must have label!
        """
        # sparsification
        mas_message = await self._extract_mas_message(mas_message=mas_message)  
        
        # add into memory
        await self.task_layer.add_task_node(mas_message.task_main)

        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,   
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            await asyncio.to_thread(
                self.main_memory.add_documents,
                [memory_doc]
            )
            # self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError('The mas_message must have label!')
        
        # finetune and merge insights
        if self.memory_size >= self._start_insights_threshold and self.memory_size % self._rounds_per_insights == 0:
            await self.insights_layer.finetune_insights(self._insights_point_num)
        if self.memory_size % 20 == 0: 
            await self.insights_layer.merge_insights() 

        # self._index_done()

    async def _retrieve_memory_raw(
        self, 
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        insight_windows: int = 10,
        threshold: float = 0.3
    ) -> tuple[list, list, list]:

        async def sort_and_filter_by_similarity(docs: list[Document], threshold: float = 0.3) -> list[tuple[Document, float]]:
            result = []
            for doc in docs:
                embedding = await self.embedding.get_embedding(doc.page_content)
                sim = await self.embedding.compute_similarity(origin_embedding, embedding)
                if sim >= threshold:
                    result.append((doc, sim))

            result.sort(key=lambda x: x[1], reverse=True)
            return result

        true_tasks_doc: list[Document] = []
        false_tasks_doc: list[Document] = []
        
        # find related tasks in task layer
        related_point_num: int = max((successful_topk + failed_topk) // 2, 1)
        task_mains: list[str] = await self.task_layer.retrieve_related_task(query_task=query_task, node_num=related_point_num, hop=self._hop)
        for task_main in task_mains:
            doc = await asyncio.to_thread(
                self.main_memory.similarity_search,
                task_main,
                1
            )
            doc = doc[0]
            # doc = self.main_memory.similarity_search(task_main, k=1)[0]

            if doc.metadata.get('label') == True:
                true_tasks_doc.append(doc)
            elif doc.metadata.get('label') == False:
                false_tasks_doc.append(doc)
            else:
                raise RuntimeError('The document object\'s metadata should have `label` attribute.')
        
        # If the specified number is not met, fill in the rest using similarity-based augmentation.
        if len(true_tasks_doc) < successful_topk:
            true_tasks_doc = await asyncio.to_thread(
                self.main_memory.similarity_search,
                query_task,
                successful_topk,
                {"label": False}
            )
            # true_tasks_doc = self.main_memory.similarity_search(
            #     query=query_task, k=successful_topk, filter={'label': True}
            # )
            for doc in true_tasks_doc:
                if doc not in true_tasks_doc:
                    true_tasks_doc.append(doc)
        
        if len(false_tasks_doc) < failed_topk:
            false_tasks_doc = await asyncio.to_thread(
                self.main_memory.similarity_search,
                query_task,
                failed_topk,
                {"label": False}
            )

            # false_tasks_doc = self.main_memory.similarity_search(
            #     query=query_task, k=failed_topk, filter={'label': False}
            # )
            for doc in false_tasks_doc:
                if doc not in false_tasks_doc:
                    false_tasks_doc.append(doc)

        # order by similarity        
        origin_embedding: list[float] = await self.embedding.get_embedding(query_task)
        true_tasks_doc_with_score = await sort_and_filter_by_similarity(true_tasks_doc, threshold)
        true_tasks_doc_with_score = true_tasks_doc_with_score[:successful_topk]
        false_tasks_doc_with_score = await sort_and_filter_by_similarity(false_tasks_doc, threshold)
        false_tasks_doc_with_score = false_tasks_doc_with_score[:failed_topk]

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc, _ in true_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc, _ in false_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        # get insights and order by relelvance
        insights_with_score = await self.insights_layer.query_insights_with_score(query_task, top_k=insight_windows)
        insights = [insight for insight, _ in insights_with_score][:insight_windows]

        return true_task_messages, false_task_messages, insights

    async def retrieve_memory(
        self, 
        query_task: str,         
        successful_topk: int = 2, 
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
        **args
    ) -> tuple[list, list, list]: 
        """Access the memory and return the results.

        Args:
            query_task (str): The task to query.
            successful_topk (int, optional): Number of successful cases to retrieve. Defaults to 2.
            failed_topk (int, optional): Number of failed cases to retrieve. Defaults to 1.
            insight_topk (int, optional): Number of insights to retrieve. Defaults to 10.
            threshold (float, optional): Similarity threshold for retrieving cases. Defaults to 0.3.

        Returns:
            tuple[list, list, list]: A tuple containing successful cases, failed cases, and insights.
        """
        
        # retrieve raw tasks
        successful_task_trajectories: list[MASMessage]
        failed_task_trajectories: list[MASMessage]
        insights: list[str]
        successful_task_trajectories, failed_task_trajectories, insights = await self._retrieve_memory_raw(
            query_task, 2*successful_topk, 2*failed_topk, 2*insight_topk, threshold)
        
        # retrieve tasks based on task relevance
        importance_score: list[float] = []
        for success_task in successful_task_trajectories:
            prompt: str = GMemoryPrompts.generative_task_user_prompt.format(
                trajectory=success_task.task_description + '\n' + success_task.task_trajectory,
                query_scenario=query_task
            )
            agent = Agent(GMemoryPrompts.generative_task_system_prompt, model='gpt-4o-mini')
            response: str = await agent.ask(prompt)
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_score.append(score)
        
        sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                           key=lambda x: x[0], reverse=True)]
        top_success_task_trajectories = sorted_success_tasks[:successful_topk]
        top_success_task_trajectories = successful_task_trajectories[:successful_topk]
        
        # directly get failed tasks
        top_fail_task_trajectories = failed_task_trajectories[:failed_topk]
        
        # directlt get insights
        top_k_insights = insights[:insight_topk]
        self.insights_cache = top_k_insights

        return top_success_task_trajectories, top_fail_task_trajectories, top_k_insights


    async def _extract_mas_message(self, mas_message: MASMessage) -> MASMessage:

        mas_message_copy: MASMessage = copy.deepcopy(mas_message)
        state_chain: StateChain = mas_message_copy.chain_of_states
        
        for state_id in reversed(range(len(state_chain))):
            if state_chain.get_state(state_id).graph.get('reward', 0) < 0:
                state_chain.pop_state(state_id)
        
        trajectory = ''
        for state in state_chain:
            trajectory += f"> {state.graph['action']}\n{state.graph['observation']}\n"
        
        if mas_message_copy.label == True:
            mas_message_copy.task_trajectory = trajectory

        
        trajectory = re.sub(r'\d+', '', trajectory)
        mas_message_copy.add_extra_field('clean_traj', trajectory)


        system_prompt = GMemoryPrompts.extract_true_traj_system_prompt
        prompt_template = GMemoryPrompts.extract_true_traj_user_prompt

        prompt: str = prompt_template.format(
            task=mas_message_copy.task_description,
            trajectory=mas_message_copy.get_extra_field('clean_traj')
        )
        agent = Agent(system_prompt, model='gpt-4o-mini')
        response: str = await agent.ask(prompt, temperature=0.1)
        mas_message_copy.add_extra_field('key_steps', response)


        if mas_message_copy.label == False:
            reason: str = await self._detect_mistakes(mas_message_copy)
            mas_message_copy.add_extra_field('fail_reason', reason)
        
        return mas_message_copy
    
    
    async def _detect_mistakes(self, mas_message: MASMessage) -> str:
        user_prompt: str = GMemoryPrompts.detect_mistakes_user_prompt.format(task=mas_message.task_description, trajectory=mas_message.get_extra_field('clean_traj'))
        agent = Agent(GMemoryPrompts.detect_mistakes_system_prompt, model='gpt-4o-mini')
        response: str = await agent.ask(user_prompt)

        return response

    def backward(self, reward: bool):

        for insight in self.insights_cache:
            self.insights_layer.backward(insight, reward=-2 if reward == False else 1)

        self.insights_cache = []
    
    @property
    def memory_size(self):
        num_records = self.main_memory.get()["ids"]
        return len(num_records)
    
    async def project_insights(self, raw_insights: list[str], role: str = None, task_traj: str = None) -> list[str]:
        """
        Projects raw insights into role-specific insights based on the given role and optionally a task trajectory.

        Args:
            raw_insights (list[str]): A list of raw insight strings.
            role (str, optional): The role to tailor the insights for. Defaults to None.
            task_traj (str, optional): A string representing the task trajectory. Defaults to None.

        Returns:
            list[str]: A list of processed insights tailored to the specified role.
        """
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        # If no role is provided, return the raw insights as they are.
        if not role:
            return raw_insights
        
        # Determine which system and user prompts to use based on whether a task trajectory is provided
        raw_insights_str = '\n'.join(raw_insights)
        if not task_traj:
            system_prompt = GMemoryPrompts.project_insights_system_prompt
            user_prompt: str = GMemoryPrompts.project_insights_user_prompt.format(
                role=role,
                insights=raw_insights_str
            )
        else:
            system_prompt = GMemoryPrompts.project_insights_with_traj_system_prompt
            user_prompt: str = GMemoryPrompts.project_insights_with_traj_user_prompt.format(
                role=role,
                insights=raw_insights_str,
                trajectory=task_traj
            )
        
        # Use the language model to generate role-specific insights
        agent = Agent(system_prompt, model='gpt-4o-mini')
        role_insights = await agent.ask(user_prompt)

        try: 
            role_insights = parse_numbered_list(role_insights)
            return role_insights
        except:
            return raw_insights

    @staticmethod
    def format_task_context(task_description: str, task_traj: str, key_steps: str = None) -> str:
        task_format: str = """
        ### Task description:   
        {task_description}    

        ### Key steps:
        {key_steps}

        ### Detailed trajectory:
        {trajectory}
        """
        return task_format.format(
            task_description=task_description,
            key_steps=key_steps,
            trajectory=task_traj
        )
    
    async def general_retrieve(self, recorder: Basic_Recorder):
        try:
            assert 'obs' in recorder.init
            task = recorder.init.get('obs')
            if 'Your task is to:' in task:
                task_main = self.task_type + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()
                task_description = task.split('Your task is to:')[1]
            else:
                task_main = task + recorder.init.get('goal','')
                task_description = recorder.init.get('goal','')
        except Exception as e:
            task_main, task_description = recorder.init.get('goal')+recorder.init.get('long_term_context')+recorder.init.get('short_term_context'), recorder.init.get('goal')

        
        # self.init_task_context(task_main, task_description) 
        
        # Retrieve successful trajectories and insights from memory
        successful_trajectories: list[MASMessage]
        insights: list[dict]
        
        successful_trajectories, _, insights = await self.retrieve_memory( # use default settings
            query_task=task_main,
            successful_topk=1,
            failed_topk=1,
            insight_topk=3,
            threshold=0
        )
        successful_shots: list[str] = [self.format_task_context(
            traj.task_description, traj.task_trajectory, traj.get_extra_field('key_steps')
        ) for traj in successful_trajectories]
        raw_rules: list[str] = [insight for insight in insights] # just not allow project for consistency/ only have one role
        return {
            'successful_shots': successful_shots,
            'useful_insights': raw_rules
        }

    async def general_update(self, recorder: Basic_Recorder):
        try:
            assert 'obs' in recorder.init
            task = recorder.init.get('obs')
            if 'Your task is to:' in task:
                task_main = self.task_type + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()
                task_description = task.split('Your task is to:')[1]
            else:
                task_main = task + recorder.init.get('goal','')
                task_description = recorder.init.get('goal','')
        except Exception as e:
            task_main, task_description = recorder.init.get('goal')+recorder.init.get('long_term_context')+recorder.init.get('short_term_context'), recorder.init.get('goal')
        # print('task main:', task_main)
        # print('task discription:', task_description)
        self.init_task_context(task_main, task_description) 
        # print('finish init store')
        for step in recorder.steps:
            # agent_message: AgentMessage = AgentMessage( #not allow agent infos for now
            #     agent_name=name,
            #     system_instruction=system_instruction,
            #     user_instruction=user_prompt,
            #     message=action,
            # )
            # self.add_agent_node(agent_message, upstream_agent_ids=[])
            try:
                assert 'obs' in step
                self.move_memory_state(step.get('action_took'), step.get('obs'), reward= step.get('scores'))
            except Exception as e:
                self.move_memory_state(step.get('action_took'), step.get('long_term_context')+step.get('short_term_context'), reward= step.get('scores'))
        # print('finish steps store')

        await self.save_task_context(label = recorder.reward == 1.0, feedback="You successfully finished this task!" if recorder.reward == 1.0 else "You failed the task.")  
        self.backward(recorder.reward == 1.0)    
        return


@dataclass
class TaskLayer:
    
    working_dir: str
    namespace: str
    task_storage: Chroma
    
    def __post_init__(self):
        self.similarity_threshold = 0.7

        self._graph_pic_save_path: str = os.path.join(self.working_dir, 'graph.png')
        self._node_match_save_path: str = os.path.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.pkl')

        if os.path.exists(self._graph_save_path):
            with open(self._graph_save_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"Graph loaded from {self._graph_save_path}")
        else:
            self.graph = nx.Graph()
            print("New empty graph created")

    async def add_task_node(self, task_main: str) -> None:
        """Add a task node to the task graph.

        Args:
            task_main (str): task name
        """
        if task_main in self.graph:
            return  

        self.graph.add_node(task_main)

        results: list[tuple[Document, float]] = await asyncio.to_thread(
                self.task_storage.similarity_search_with_score,
                task_main,
                10
            )

        # results: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(
        #     query=task_main,
        #     k=10 
        # )
        
        for doc, distance in results:
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue  

            neighbor = doc.page_content

            if neighbor not in self.graph:
                self.graph.add_node(neighbor)

            self.graph.add_edge(task_main, neighbor, weight=similarity) 
        
        self._index_done()
 
    async def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list[str]:
        """
        Retrieve related tasks from the graph based on similarity and local neighborhood expansion.

        Args:
            query_task (str): The task used as the query input.
            node_num (int): The number of top similar tasks to retrieve based on similarity scores.
            hop (int, optional): The number of hops used to expand the neighborhood in the graph. Defaults to 1.

        Returns:
            list[str]: A list of related task nodes, including top similar tasks and their neighbors within the given hop.
        """
        tasks: list[tuple[Document, float]] = await asyncio.to_thread(
                self.task_storage.similarity_search_with_score,
                query_task,
                node_num
        )

        # tasks: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(query=query_task, k=node_num)
        top_nodes = [doc[0].page_content for doc in tasks]

        related_nodes = set(top_nodes)
        for node in top_nodes:
            neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
            related_nodes.update(neighbours)
        return list(related_nodes)
    
    async def cluster_tasks(self) -> None:
        """
        Perform clustering on tasks in the graph using their embeddings and assign cluster IDs.

        This method extracts all nodes from the graph, computes embeddings for each node using the
        task storage's embedding function, and applies the FINCH clustering algorithm with cosine similarity.
        """
        nodes = list(self.graph.nodes)

        embeddings = []
        valid_nodes = []

        for node in nodes:
            embedding = await asyncio.to_thread(
                self.task_storage._embedding_function.embed_query,
                node
            )
            # embedding = self.task_storage._embedding_function.embed_query(node)  
            if embedding is not None:
                embeddings.append(embedding)
                valid_nodes.append(node)

        X = np.vstack(embeddings)

        try: 
            c, _, _ = FINCH(X) #metric='cosine'
            labels = c[:,-1]
        except Exception as e:   
            print(f"FINCH clustering failed: {e}")
            labels = np.zeros(len(valid_nodes), dtype=int)

        for node, label in zip(valid_nodes, labels):
            self.graph.nodes[node]['cluster_id'] = int(label)
        self._index_done()

    def _index_done(self) -> None:
        
        with open(self._graph_save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def __iter__(self) -> Iterable[tuple[str, int]]: 
        return ((node, self.graph.nodes[node]['cluster_id']) for node in self.graph.nodes)

@dataclass
class InsightsManager:

    working_dir: str
    namespace: str
    task_storage: Chroma
    task_layer: TaskLayer
    def __post_init__(self):
        self.persist_file: str = os.path.join(self.working_dir,f'{self.namespace}.json')
        self.insights_memory: list[dict] = load_json(self.persist_file) or []
        

    async def query_insights_with_score(self, task_query: str, top_k: int = None) -> list[tuple[str, float]]:

        SUCC_NUM, FAIL_NUM = 4, 2

        related_successful_tasks, related_failed_tasks = await self._retrieve_memory(task_query, successful_topk=SUCC_NUM, failed_topk=FAIL_NUM)
        task_mains: list[str] = [task.task_main for task in related_successful_tasks + related_failed_tasks]
        task_mains.append(task_query)
        insights_score = defaultdict(float)
        for task_main in task_mains:
            _, related_insights = self._find_related_insights(task_mains=[task_main])
            for insight in related_insights:
                insights_score[insight.get('rule')] += 1  

        sorted_insights = sorted(insights_score.items(), key=lambda x: x[1], reverse=True) 
        if top_k is not None:
            sorted_insights = sorted_insights[:top_k]
        return sorted_insights
    
    async def merge_insights(self) -> None:

        await self.task_layer.cluster_tasks()
        
        label_tasks: dict[int, list[str]] = {}
        for task_main, label_id in self.task_layer:
            if label_id is None:
                raise RuntimeError('Label id should not be none.')
            if label_id not in label_tasks.keys():
                label_tasks[label_id] = [task_main]
            else:
                label_tasks[label_id].append(task_main)
        
        merged_label_rules: dict[int, list[str]] = {}
        for task_type, related_task_mains in label_tasks.items():
            related_ids, related_insights = self._find_related_insights(task_mains=related_task_mains)
            related_rules: list[str] = [insight['rule'] for insight in related_insights]
            merged_rules: list[str] = await self._merge_rules(related_rules)
            merged_label_rules[task_type] = merged_rules

            print('------- Merge Insights -------')
            # print(f'Task type: {task_type}')
            # related = '\n'.join(related_rules)
            # print(f"Origin rules: \n{related}")
            # merged = '\n'.join(merged_rules)
            # print(f"Merged rules: \n{merged}")
            
        self.insights_memory.clear()

        for label, related_rules in merged_label_rules.items():
            related_task_mains = label_tasks.get(label)
            if related_task_mains is None:
                raise RuntimeError('Inconsistency in `label`')
            
            for rule in related_rules:
                insight: dict = {
                    'rule': rule,
                    'score': 2,          
                    'positive_correlation_tasks': list(related_task_mains),
                    'negative_correlation_tasks': list()
                }
                self.insights_memory.append(insight)
        
        self._index_done()

    async def _merge_rules(self, rules: list[str]) -> list[str]:
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        merged_rules = []
        batch_size = 10

        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            actual_num: int = len(batch) // 3  

            user_prompt = GMemoryPrompts.merge_rules_user_prompt.format(
                current_rules='\n'.join(batch),
                limited_number=actual_num//3
            )
            agent = Agent(GMemoryPrompts.merge_rules_system_prompt, model='gpt-4o-mini')
            raw_merged_rules = await agent.ask(user_prompt)
            merged_rules.extend(parse_numbered_list(raw_merged_rules))

        return merged_rules

    def backward(self, insight: str, reward: float):
        
        for inner_insight in self.insights_memory:
            if insight in inner_insight['rule']:
                inner_insight['score'] += reward

        self.clear_insights()
        self._index_done()

    def clear_insights(self):
        self.insights_memory = [self.insights_memory[i] for i in range(len(self.insights_memory)) 
                        if self.insights_memory[i]['score'] > 0] 

    async def _retrieve_memory(
        self,
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1
    ) -> tuple[list[MASMessage], list[MASMessage]]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = await asyncio.to_thread(
                self.task_storage.similarity_search_with_score,
                query_task,
                successful_topk,
                {'label': True}
            )
            # true_tasks_doc = self.task_storage.similarity_search_with_score(
            #     query=query_task, k=successful_topk, filter={'label': True}
            # )
        if failed_topk != 0:
            false_tasks_doc = await asyncio.to_thread(
                self.task_storage.similarity_search_with_score,
                query_task,
                failed_topk,
                {'label': False}
            )
            # false_tasks_doc = self.task_storage.similarity_search_with_score(
            #     query=query_task, k=failed_topk, filter={'label': False}
            # )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages
    
    @property
    def task_size(self):
        num_records = self.task_storage.get()["ids"]
        return len(num_records)
    
    def _find_related_insights(
        self,
        task_mains: list[str],
        threshold: float = 1
    ) -> tuple[list[int], list[dict]]:

        rule_set: list[tuple[dict, int, int]] = []  # (rule, score, index)

        for idx, rule in enumerate(self.insights_memory):
            score: int = sum(task in rule.get('positive_correlation_tasks', []) for task in task_mains)
            if score >= threshold:
                rule_set.append((rule, score, idx))

        rule_set.sort(key=lambda x: x[1], reverse=True)

        rule_indices = [item[2] for item in rule_set]
        sorted_rules = [item[0] for item in rule_set]

        return rule_indices, sorted_rules
    async def finetune_insights(self, num_points: int):

        SUCCESS_TASK_NUM, FAIL_TASK_NUM = 3, 1
        # all_ids = await asyncio.to_thread(
        #         self.task_storage.get
        #     )
        # all_ids = all_ids['ids']
        all_ids = self.task_storage.get()['ids']
        for _ in range(num_points):  

            random_id = random.choice(all_ids)
            random_entry = await asyncio.to_thread(
                self.task_storage.get,
                [random_id]
            )

            # random_entry = self.task_storage.get(ids=[random_id])
            if 'metadatas' in random_entry and random_entry['metadatas']:
                random_metadata = random_entry['metadatas'][0]  
            else:
                raise RuntimeError('Incomplete data.')
            mas_message: MASMessage = MASMessage.from_dict(random_metadata)


            true_trajs, false_trajs = await self._retrieve_memory(
                query_task=mas_message.task_main, successful_topk=SUCCESS_TASK_NUM, failed_topk=FAIL_TASK_NUM
            )
            if mas_message.label == True:
                true_trajs.append(mas_message)
            else:
                false_trajs.append(mas_message)
            all_task_mains: list[str] = [traj.task_main for traj in true_trajs + false_trajs]

            related_insight_ids, _ = self._find_related_insights(all_task_mains, len(all_task_mains) / 2)
            await self._finetune_insights(true_trajs, false_trajs, related_insight_ids)
        
        self.clear_insights()
        self._index_done()
    async def _finetune_insights(
        self,
        successful_task_trajectories: list[MASMessage],
        failed_task_trajectories: list[MASMessage],
        insight_ids: list[int]
    ) -> None:

        def map_operations(origin_operations: list[tuple]) -> list[tuple]:
            processed_operations: list[tuple] = []
            for (operation, text) in origin_operations:
                res: list = operation.split(' ')

                if len(res) == 2:
                    if len(insight_ids) == 0:     
                        continue
                    insight_id: int = int(res[1]) - 1
                    if insight_id >= len(insight_ids) or insight_id < 0:
                        continue
                    
                    res[1] = str(insight_ids[insight_id] + 1)   
                    operation: str = ' '.join(res)
                processed_operations.append((operation, text))
            
            return processed_operations

        rule_list: list[dict] = [self.insights_memory[i] for i in insight_ids]

        compare_pairs: list[tuple[MASMessage, MASMessage]] = []
        for id, fail_task in enumerate(failed_task_trajectories):
            if id >= len(successful_task_trajectories):
                break
            success_task = successful_task_trajectories[id]
            compare_pairs.append((success_task, fail_task))
        
        successful_task_chunks: list[list[MASMessage]] = random_divide_list(successful_task_trajectories, 5) 
        
        MAX_RULE_THRESHOLD: int = 10
        suffix: str = GMemoryPrompts.finetune_insights_suffix['full'] if len(self.insights_memory) > MAX_RULE_THRESHOLD \
                      else GMemoryPrompts.finetune_insights_suffix['not_full']


        print('--------------- Finetune Insights ---------------')
        for pair in compare_pairs:
            compare_prompts: list[Dict] = self._build_comparative_prompts(pair[0], pair[1], rule_list)
            compare_prompts[0]['content'] += suffix
            agent = Agent('', model='gpt-4o-mini')
            response: str = await agent.ask(compare_prompts, with_full_msg = True)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            self._update_rules(
                [pair[0].task_main, pair[1].task_main], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            # print(compare_prompts[0]['role'] + compare_prompts[0]['content'] + '\n\n' + compare_prompts[1]['role'] + compare_prompts[1]['content'])
            # print(response)
            # print('\n---------------\n')

        for chunk in successful_task_chunks:
            success_prompts: list[Dict] = self._build_success_prompts(chunk, rule_list) 
            success_prompts[0]['content'] += suffix
            agent = Agent('', model='gpt-4o-mini')
            response: str = await agent.ask(success_prompts, with_full_msg = True)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            task_mains: list[str] = [traj.task_main for traj in chunk]
            self._update_rules(
                task_mains, 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            # print(success_prompts[0]['role'] + success_prompts[0]['content'] + '\n\n' + success_prompts[1]['role'] + success_prompts[1]['content'])
            # print(response)
            # print('\n---------------\n')
        
        self.clear_insights()
        self._index_done()

    def _index_done(self):
        write_json(self.insights_memory, self.persist_file)

    def _build_comparative_prompts(self, true_traj: MASMessage, false_traj: MASMessage, insights: list[dict]) -> list[Dict]:
        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        prompt = GMemoryPrompts.critique_compare_rules_user_prompt.format(   
            task1=true_traj.task_description,
            task1_trajectory=true_traj.task_trajectory,   
            task2=false_traj.task_description,
            task2_trajectory=false_traj.task_trajectory,
            fail_reason=false_traj.get_extra_field('fail_reason'),
            existing_rules=rule_text
        )

        return [{'role':'system', 'content': GMemoryPrompts.critique_compare_rules_system_prompt}, {'role': 'user', 'content':prompt}] 
    
    def _build_success_prompts(
        self,
        success_trajectories: Iterable[MASMessage],
        insights: list[dict],
    ) -> list[Dict]:

        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        history: list[str] = [f'task{i}:\n' + task.task_description + task.get_extra_field('key_steps') for i, task in enumerate(success_trajectories)]
        prompt = GMemoryPrompts.critique_success_rules_user_prompt.format(
            success_history='\n'.join(history),
            existing_rules=rule_text
        )
        return [{'role':'system', 'content': GMemoryPrompts.critique_success_rules_system_prompt}, {'role': 'user', 'content': prompt}] 
    
    def _parse_rules(self, llm_text):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):

                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return(res)
    
    def _update_rules(
        self,
        relative_tasks: list[str],
        operations: list[tuple[str, str]], 
        max_rules_num: int = 10
    ) -> None:

        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':    
                if self._is_existing_rule(operation_rule_text): 
                    delete_indices.append(i)
                    
            elif operation_type == 'EDIT':   
                if self._is_existing_rule(operation_rule_text): 
                    rule_num: int = self._retrieve_rule_index(operation_rule_text)
                    operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)   

                elif (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
                        
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':  
                if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
            
            else: 
                delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] 
        

        list_full: bool = len(self.insights_memory) >= max_rules_num  
        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: 
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    remove_strength = 3 if list_full else 1
                    rule_data['score'] -= remove_strength
                    rule_data['negative_correlation_tasks'] = list(set(rule_data['negative_correlation_tasks'] + relative_tasks))  

                elif operation_type == 'AGREE':
                    rule_index: int = self._retrieve_rule_index(operation_rule_text) 
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'EDIT': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['rule'] = operation_rule_text
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'ADD': 
                    meta_data: dict = {
                        'rule': operation_rule_text,
                        'score': 2,         
                        'positive_correlation_tasks': list(relative_tasks),
                        'negative_correlation_tasks': list()
                    }
                    self.insights_memory.append(meta_data)

    def _is_existing_rule(self, operation_rule_text: str) -> bool:

        for insight in self.insights_memory:
            if insight['rule'] in operation_rule_text:
                return True
        return False
    
    def _retrieve_rule_index(self, operation_rule_text: str) -> int:

        for idx, insight in enumerate(self.insights_memory):
            if insight['rule'] in operation_rule_text:
                return idx
        return -1