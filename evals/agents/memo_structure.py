from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass
from eval_envs.base_envs import Basic_Recorder

@dataclass
class Sub_memo_layer(ABC):
    """
    Abstract class for retrieve/update sub-function
    """
    layer_intro: str = "Introduction of the structure of current defined Database(if any), corresponding Update and Retrieve method."
    database: Optional[Any] = None

    @abstractmethod
    async def retrieve(self, **kwargs):
        """
        The retrieve function of current layer. 
        """
        pass

    @abstractmethod
    async def update(self, **kwargs):
        """
        The update function of current layer. 
        """
        pass

class MemoStructure(ABC):

    def __init__(self):
        self.database: Optional[Any] = None
        
    # -------- Pipeline Runner --------
    @abstractmethod
    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict: 
        """
        The general retrieve method, use the retrieve function in each layer, determine order and input-output usage by yourself.
        """
        pass

    @abstractmethod
    async def general_update(self, recorder: Basic_Recorder) -> None:  
        """
        The general update method, use the update function in each layer, determine order and input-output usage by yourself.
        """
        pass