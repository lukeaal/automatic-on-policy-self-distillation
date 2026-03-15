from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class CompletionResponse(BaseModel):
    text: str

class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, prompts: List[str], **kwargs) -> List[CompletionResponse]:
        """ takes in list of prompts and return list of completion responses """
        # TODO
        pass