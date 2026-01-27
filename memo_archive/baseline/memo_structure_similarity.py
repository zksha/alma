from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from typing import Dict, Optional, List

class SimilarityMemory(MemoStructure):
    def __init__(self):
        super().__init__()
        self.sim_db = {}

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
        # ----
        init_dict = recorder.init
        init_obs = ''
        for key, val in init_dict.items():
            init_obs += f'\n<{key}>{val}</{key}>'
        best_candidate = await best_match_by_embedding(init_obs, list(self.sim_db.keys()))
        memo = {
            'retrieved_similar_tasks':self.sim_db[best_candidate]
        }
        return memo

    async def general_update(self, recorder: Basic_Recorder) -> None:
        init_dict = recorder.init
        init_obs = ''
        for key, val in init_dict.items():
            init_obs += f'\n<{key}>{val}</{key}>'
        steps = recorder.steps
        reward = getattr(recorder, 'reward', 0.0)
        self.sim_db[init_obs] = {
            "initial_env":init_dict,
            "steps": steps,
            "reward": reward
        }
        return 