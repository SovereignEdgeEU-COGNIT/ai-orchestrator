from collections.abc import Collection, Mapping
from typing import Any, Optional, List, Dict, Tuple, Set
from dataclasses import dataclass

from mapper import Mapper
from model import Allocation, HostCapacity, VMRequirements

@dataclass
class HostState:
    """Represents the current state of a host including its available resources."""
    id: int
    available_memory: float
    available_cpu: float
    total_memory: float
    total_cpu: float
    cluster_id: int = 0

class BestFitMapper(Mapper):
    """
    Implements the Best Fit Decreasing algorithm for VM placement.
    This algorithm considers resource intensity of VMs and matches them with hosts
    that have corresponding resource capacities. It profiles VMs as CPU-intensive,
    memory-intensive, or balanced, and places them accordingly.

    Note
    ----
    This algorithm assumes that the provided vm_requirements list contains ONLY the VMs
    that need to be placed. It does not perform any filtering or validation to check if
    VMs are already placed or if they need placement. The user must ensure they provide
    only the VMs that require placement.
    """
    
    # Resource utilization threshold (80%)
    RESOURCE_THRESHOLD = 0.8
    
    # Percentage difference required to consider a VM resource-intensive (50%)
    # If one resource is 50% higher than the other, the VM is considered intensive in that resource
    RESOURCE_INTENSITY_THRESHOLD = 0.5
    
    # Waste calculation weights
    WASTE_PRIMARY_WEIGHT = 0.7
    WASTE_SECONDARY_WEIGHT = 0.3
    
    def __init__(
        self,
        current_placement: Mapping[int, int],
        vm_requirements: Collection[VMRequirements],
        host_capacities: Collection[HostCapacity],
        criteria: Any,
        **kwargs
    ) -> None:
        """
        Initialize the Best Fit Decreasing mapper.

        Parameters
        ----------
        current_placement : Mapping[int, int]
            Current allocation mapping of VMs to Hosts.
        vm_requirements : Collection[VMRequirements]
            Collection of VM requirements including memory and CPU needs.
        host_capacities : Collection[HostCapacity]
            Collection of host capacities including available resources.
        criteria : Any
            Mapping criteria to be used (not used in this implementation).
        **kwargs : dict
            Additional keyword arguments.
        """
        self.current_placement = current_placement
        self.vm_requirements = list(vm_requirements)
        self.host_capacities = list(host_capacities)
        self.criteria = criteria
        
        # Initialize host states with both available and total capacities
        self.host_states: Dict[int, HostState] = {
            host.id: HostState(
                id=host.id,
                available_memory=host.memory.free,
                available_cpu=host.cpu.free,
                total_memory=host.memory.total,
                total_cpu=host.cpu.total,
                cluster_id=host.cluster_id
            )
            for host in host_capacities
        }
        
        # Store valid host IDs for validation
        self.valid_host_ids: Set[int] = {host.id for host in host_capacities}
        
        # Initialize solution storage
        self.solutions: List[Dict[int, Optional[Allocation]]] = []
        
    def _profile_vm(self, vm: VMRequirements) -> Tuple[str, float]:
        """
        Profile a VM based on its resource requirements to determine if it's
        CPU-intensive, memory-intensive, or balanced.

        Parameters
        ----------
        vm : VMRequirements
            The VM requirements to profile.

        Returns
        -------
        Tuple[str, float]
            A tuple containing the VM profile ('cpu', 'memory', or 'balanced')
            and a normalized score for sorting.
        """
        # Calculate resource ratios
        memory_ratio = vm.memory / max(vm.memory for vm in self.vm_requirements)
        cpu_ratio = vm.cpu_ratio / max(vm.cpu_ratio for vm in self.vm_requirements)
        
        # Determine VM profile based on percentage difference between resources
        # If one resource is {RESOURCE_INTENSITY_THRESHOLD}% higher than the other,
        # the VM is considered intensive in that resource
        if cpu_ratio > memory_ratio * (1 + self.RESOURCE_INTENSITY_THRESHOLD):
            return 'cpu', cpu_ratio
        elif memory_ratio > cpu_ratio * (1 + self.RESOURCE_INTENSITY_THRESHOLD):
            return 'memory', memory_ratio
        else:
            return 'balanced', (cpu_ratio + memory_ratio) / 2
        
    def _calculate_vm_score(self, vm: VMRequirements) -> float:
        """
        Calculate a normalized score for VM based on its resource requirements.
        This score is used for sorting VMs in descending order.

        Parameters
        ----------
        vm : VMRequirements
            The VM requirements to score.

        Returns
        -------
        float
            A normalized score combining memory and CPU requirements.
        """
        _, score = self._profile_vm(vm)
        return score
        
    def _can_fit(self, vm: VMRequirements, host_state: HostState) -> bool:
        """
        Check if a VM can fit on a host based on resource requirements.
        Maintains a safety threshold of 80% for both memory and CPU resources.

        Parameters
        ----------
        vm : VMRequirements
            The VM requirements to check.
        host_state : HostState
            The current state of the host.

        Returns
        -------
        bool
            True if the VM can fit on the host while maintaining safety thresholds,
            False otherwise.
        """
        # Calculate projected resource utilization after placing the VM
        projected_memory_usage = (host_state.total_memory - host_state.available_memory + vm.memory) / host_state.total_memory
        projected_cpu_usage = (host_state.total_cpu - host_state.available_cpu + vm.cpu_ratio) / host_state.total_cpu
        
        # Check if projected usage exceeds the safety threshold
        if projected_memory_usage > self.RESOURCE_THRESHOLD or projected_cpu_usage > self.RESOURCE_THRESHOLD:
            return False
            
        # Check if VM requirements can be met with available resources
        return (vm.memory <= host_state.available_memory and 
                vm.cpu_ratio <= host_state.available_cpu)
    
    def _calculate_waste(self, vm: VMRequirements, host_state: HostState) -> float:
        """
        Calculate the waste of resources if placing VM on host.
        Considers the VM's resource profile and host's resource capacities.

        Parameters
        ----------
        vm : VMRequirements
            The VM requirements.
        host_state : HostState
            The current state of the host.

        Returns
        -------
        float
            The amount of wasted resources (normalized sum of memory and CPU waste).
        """
        vm_profile, _ = self._profile_vm(vm)
        
        # Calculate normalized waste for each resource
        memory_waste = (host_state.available_memory - vm.memory) / host_state.total_memory
        cpu_waste = (host_state.available_cpu - vm.cpu_ratio) / host_state.total_cpu
        
        # Adjust weights based on VM profile and host capacity
        if vm_profile == 'cpu':
            # For CPU-intensive VMs, prioritize CPU waste
            return self.WASTE_PRIMARY_WEIGHT * cpu_waste + self.WASTE_SECONDARY_WEIGHT * memory_waste
        elif vm_profile == 'memory':
            # For memory-intensive VMs, prioritize memory waste
            return self.WASTE_PRIMARY_WEIGHT * memory_waste + self.WASTE_SECONDARY_WEIGHT * cpu_waste
        else:
            # For balanced VMs, consider both resources equally
            return (cpu_waste + memory_waste) / 2
    
    def _validate_host_ids(self, vm: VMRequirements) -> None:
        """
        Validate that all host IDs specified for a VM exist in the available hosts.

        Parameters
        ----------
        vm : VMRequirements
            The VM requirements to validate.

        Raises
        ------
        ValueError
            If VM has no host_ids specified.
        KeyError
            If any of the specified host IDs don't exist in available hosts.
        """
        if vm.host_ids is None:
            raise ValueError(f"VM {vm.id} does not have host_ids specified. Each VM must explicitly specify which hosts it can be placed on.")
        
        invalid_hosts = vm.host_ids - self.valid_host_ids
        if invalid_hosts:
            raise KeyError(f"VM {vm.id} specifies non-existent host IDs: {invalid_hosts}. Valid host IDs are: {self.valid_host_ids}")
    
    def map(self) -> None:
        """
        Solve the VM placement using Best Fit Decreasing algorithm.
        VMs are sorted by their resource intensity and matched with appropriate hosts.
        Only considers hosts that are specified in VM's host_ids.

        Note
        ----
        This method assumes that vm_requirements contains only the VMs that need
        placement. It does not check whether VMs are already placed or need placement.
        The user must pre-filter the VM list to include only VMs requiring placement.

        Raises
        ------
        ValueError
            If any VM does not have host_ids specified.
        KeyError
            If any VM specifies non-existent host IDs.
        """
        # Sort VMs by combined resource score in descending order
        sorted_vms = sorted(
            self.vm_requirements,
            key=self._calculate_vm_score,
            reverse=True
        )
        
        # Initialize solution
        solution: Dict[int, Optional[Allocation]] = {}
        
        # Try to place each VM
        for vm in sorted_vms:
            # Validate host IDs before attempting placement
            self._validate_host_ids(vm)
            
            best_fit_host = None
            min_waste = float('inf')
            
            # Get the list of applicable hosts for this VM
            applicable_hosts = [self.host_states[host_id] for host_id in vm.host_ids]
            
            # Find the best fitting host among applicable hosts
            for host_state in applicable_hosts:
                if self._can_fit(vm, host_state):
                    waste = self._calculate_waste(vm, host_state)
                    if waste < min_waste:
                        min_waste = waste
                        best_fit_host = host_state
            
            if best_fit_host is not None:
                # Place VM on the best fitting host
                solution[vm.id] = Allocation(vm_id=vm.id, host_id=best_fit_host.id)
                
                # Update host state
                self.host_states[best_fit_host.id].available_memory -= vm.memory
                self.host_states[best_fit_host.id].available_cpu -= vm.cpu_ratio
            else:
                # No suitable host found
                solution[vm.id] = None
        
        self.solutions = [solution]
    
    def placements(
        self, top_k: int = 1
    ) -> list[dict[int, Optional[Allocation]]]:
        """
        Get the top K placement solutions.

        Parameters
        ----------
        top_k : int, optional
            Number of top solutions to return, by default 1

        Returns
        -------
        list[dict[int, Optional[Allocation]]]
            List of the top K placement solutions.
        """
        return self.solutions[:top_k] 