from abc import ABC, abstractmethod
import math

# Interface class
class ResourceEstimatorInterface(ABC):

    @abstractmethod
    def estimate(self, requests_per_second: int, waiting_functions: list[float], busy_vms: list[float], idle_vms: int):
        """
        Calculate the optimal number of VMs needed to handle the load based on current observation REACTIVELY.
        """
        pass
    
    @abstractmethod
    def digest(self, dataset_file_path: str):
        """
        Trains a model based on the dataset provided. The model learns from the historical data.
        """
        pass
    
    @abstractmethod
    def predict(self, requests_per_second: int, waiting_functions: list[float], busy_vms: list[float], idle_vms: int):
        """
        Calculate the optimal number of VMs needed to handle the load based on current and past observations PROACTIVELY. 
        It uses underlying models to predict the future load and give the optimal number of VMs.
        """
        pass
    
    

# VMManager class implementing the interface
class VMEstimator(ResourceEstimatorInterface):
    
    def __init__(self):
        self.model = []
        # Threshold in seconds for VMs that will soon be free
        self.soon_free_threshold = 1.0

    def estimate(self, requests_per_second, waiting_functions, busy_vms, idle_vms):
        """
        Calculate the optimal number of VMs required to handle incoming and waiting functions.
        
        Parameters:
            - requests_per_second (int): Number of incoming function requests per second.
            - waiting_functions (list of floats): List of execution times for waiting functions.
            - busy_vms (list of floats): List of remaining execution times for busy VMs.
            - idle_vms (int): Number of idle VMs in the cluster.

        Returns:
            - optimal_vms (int): Total number of VMs required.
        """
        # Calculate required VMs for incoming requests
        required_vms_for_requests = math.ceil(requests_per_second)

        # Calculate required VMs for waiting functions based on their execution times
        required_vms_for_waiting = len(waiting_functions)
        
        # Total required VMs (incoming + waiting)
        total_required_vms = required_vms_for_requests + required_vms_for_waiting

        # Estimate the number of busy VMs that will soon be free
        soon_free_vms = sum(1 for time in busy_vms if time <= self.soon_free_threshold)


        # Subtract currently available resources (idle + soon-to-be-free VMs)
        current_available_vms = idle_vms + soon_free_vms

        if total_required_vms > current_available_vms:
            optimal_vms = total_required_vms + (len(busy_vms) - soon_free_vms)
        else:
            optimal_vms = total_required_vms + len(busy_vms)

        return optimal_vms
    
    def digest(self, dataset_file_path):
        """
        Train a model based on the dataset provided.
        
        Parameters:
            - dataset_file_path (str): Path to the dataset file.
        """
        # Load the dataset and train the model
        pass
    
    def predict(self, requests_per_second, waiting_functions, busy_vms, idle_vms):
        """
        Predict the optimal number of VMs needed to handle the load based on trained model's predictions and current observations.
        """
        # Use the trained model to predict the optimal number of VMs
        pass

# Example usage
if __name__ == '__main__':
    # Input parameters
    requests_per_second = 5  # Incoming requests per second
    waiting_functions = [2.0, 1.5, 3.0]  # Execution times for waiting functions
    busy_vms = [1.0, 2.5]  # Remaining execution times for busy VMs
    idle_vms = 2  # Number of idle VMs
    
    # Create an instance of VMEstimator
    vm_manager = VMEstimator()
    
    # Calculate optimal number of VMs
    optimal_vms = vm_manager.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
    
    print(f'Optimal number of VMs needed: {optimal_vms}')
