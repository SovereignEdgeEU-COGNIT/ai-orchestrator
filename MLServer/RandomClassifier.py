import random
from typing import List
from ClassifierInterface import ClassifierInterface


class RandomClassifier(ClassifierInterface):
    @staticmethod
    def get_name() -> str:
        return "RandomClassifier"
    
    def get_output_size(self) -> int:
        return 5

    def initialize(self):
        pass

    def predict(self, vm_id) -> List[float]:
        return [random.random() for _ in range(5)]

