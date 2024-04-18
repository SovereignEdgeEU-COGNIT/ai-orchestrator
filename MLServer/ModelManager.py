

from typing import Dict
from ClassifierInterface import ClassifierInterface
from enum import Enum

class ModelTypes(Enum):
    SCHEDULER = "scheduler"
    CLASSIFIER = "classifier"

class ModelManager:
    def __init__(self):
        self.models = {
            ModelTypes.SCHEDULER: Dict[str, MLModelInterface],
            ModelTypes.CLASSIFIER: Dict[str, MLModelInterface],
        }
        self.selected_models = {
            ModelTypes.SCHEDULER: None,
            ModelTypes.CLASSIFIER: None,
        }

    def add_model(self, model_type: ModelTypes, model_instance: MLModelInterface):
        if model_type in self.models:
            model_instance.initialize()
            self.models[model_type][model_instance.get_name()] = model_instance
        else:
            raise ValueError("Invalid model type specified.")

    def set_model(self, model_type: ModelTypes, model_name: str):
        if model_type in self.models:
            self.selected_models[model_type] = self.models[model_type][model_name]
        else:
            raise ValueError("Invalid model type specified.")

    def predict(self, model_type, vm_id: int) -> int:
        
        if model_type not in self.selected_models:
            raise ValueError("Invalid model type specified")
        if self.selected_models[model_type] is None:
            raise ValueError("No model selected")
        
        return self.selected_models[model_type].predict(vm_id)
        
            
