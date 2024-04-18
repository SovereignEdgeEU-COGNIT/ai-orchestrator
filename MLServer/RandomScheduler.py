

import random
from typing import List
from ClassifierInterface import ClassifierInterface
from SchedulerInterface import SchedulerInterface


class RandomScheduler(SchedulerInterface):
    @staticmethod
    def get_name() -> str:
        return "RandomScheduler"

    def initialize(self):
        pass
    
    def set_classifier(self, classifier: ClassifierInterface):
        pass

    def predict(self, vm_id: int, host_ids: List[int]) -> int:
        return host_ids[random.randint(0, len(host_ids) - 1)]