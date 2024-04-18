

from abc import ABC, abstractmethod
from typing import List

from ClassifierInterface import ClassifierInterface


class SchedulerInterface(ABC):
    @abstractmethod
    def get_name() -> str:
        pass

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def set_classifier(self, classifier: ClassifierInterface):
        pass

    @abstractmethod
    def predict(self, vm_id: int, host_ids: List[int]) -> int:
        pass