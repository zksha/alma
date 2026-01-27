from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from typing import Dict, Optional, List

class SimilarityMemory(MemoStructure):
    def __init__(self):
        super().__init__()

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        return {}

    async def general_update(self, recorder: Basic_Recorder) -> None:
        return 