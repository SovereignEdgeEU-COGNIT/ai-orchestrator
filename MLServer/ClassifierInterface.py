from abc import ABC, abstractmethod
from typing import List

class ClassifierInterface(ABC):
    @abstractmethod
    def get_name() -> str:
        pass
    
    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self, vm_id: int) -> List[float]:
        pass
