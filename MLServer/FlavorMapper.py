

from typing import Dict, List, Tuple


class FlavorMapper:
    def __init__(self, resource_length: int):
        # Each flavor now maps to a tuple containing two elements:
        # 1. A list of the sum of resources for the flavor
        # 2. A count of the number of VMs of that flavor
        self.flavors: Dict[str, Tuple[List[float], int]] = {}
        self.resource_length = resource_length
        
    def update_flavor(self, flavor: str, resources: List[float]):
        # If the flavor is not in the dictionary, initialize it
        if flavor not in self.flavors:
            self.flavors[flavor] = ([0.0] * self.resource_length, 0)
        
        # Get the current sum of resources and the count of VMs for the flavor
        current_sum, count = self.flavors[flavor]
        
        # Update the sum of resources by adding the new resources
        updated_sum = [current + new for current, new in zip(current_sum, resources)]
        
        # Increment the count of VMs
        updated_count = count + 1
        
        # Update the flavor's entry with the new sum and count
        self.flavors[flavor] = (updated_sum, updated_count)
    
    def get_flavor(self, flavor: str) -> List[float]:
        # Return the average resources for a given flavor
        if flavor in self.flavors:
            sum_resources, count = self.flavors[flavor]
            # Calculate the average for each resource
            average_resources = [total / count for total in sum_resources]
            return average_resources
        else:
            # Return a list of equal length to the resources, filled with 1 / length
            return [1 / self.resource_length] * self.resource_length