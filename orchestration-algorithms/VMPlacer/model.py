from dataclasses import dataclass, field
import enum
from collections.abc import Collection
from typing import Optional, Union

@dataclass(frozen=True)
class Allocation:
    """
    Represents a VM-to-host allocation mapping.

    Parameters
    ----------
    vm_id : int
        Unique identifier for the virtual machine.
    host_id : int
        Unique identifier for the host machine.
    """
    vm_id: int
    host_id: int

@dataclass(frozen=True)
class Capacity:
    """
    Represents resource capacity with total and used amounts.

    Parameters
    ----------
    total : Union[float, int]
        Total available capacity of the resource.
    usage : Union[float, int]
        Current usage of the resource.

    Properties
    ----------
    free : Union[float, int]
        Remaining available capacity (total - usage).
    """
    total: Union[float, int]
    usage: Union[float, int]

    @property
    def free(self) -> Union[float, int]:
        """
        Calculate remaining available capacity.

        Returns
        -------
        Union[float, int]
            The amount of unused capacity.
        """
        return self.total - self.usage

@dataclass(frozen=True)
class HostCapacity:
    """
    Represents the resource capacities of a host machine.

    Parameters
    ----------
    id : int
        Unique identifier for the host.
    memory : Capacity
        Memory capacity information of the host.
    cpu : Capacity
        CPU capacity information of the host.
    cluster_id : int, optional
        Identifier for the cluster this host belongs to, by default 0.
    """
    id: int
    memory: Capacity
    cpu: Capacity
    cluster_id: int = 0

class VMState(enum.Enum):
    """
    Enumeration of possible virtual machine states.

    Attributes
    ----------
    PENDING : str
        VM is waiting to be scheduled.
    RESCHED : str
        VM needs to be rescheduled.
    RUNNING : str
        VM is currently running.
    POWEROFF : str
        VM is powered off.
    """
    PENDING = 'pending'
    RESCHED = 'resched'
    RUNNING = 'running'
    POWEROFF = 'poweroff'

@dataclass(frozen=True)
class VMRequirements:
    """
    Represents the resource requirements and current state of a virtual machine.

    Parameters
    ----------
    id : int
        Unique identifier for the VM.
    state : Optional[VMState]
        Current state of the VM (pending, running, etc.).
    memory : int
        Requested memory allocation for the VM.
    memory_usage : int
        Actual monitored memory usage of the VM.
    cpu_ratio : float
        Requested CPU allocation ratio for the VM.
    cpu_usage : float
        Actual monitored CPU usage of the VM, defaults to NaN.
    host_ids : Optional[set[int]]
        Set of possible host IDs where this VM can be allocated, optional.

    """
    id: int
    state: Optional[VMState]
    memory: int
    memory_usage: int
    cpu_ratio: float
    cpu_usage: float = float('nan')
    host_ids: Optional[set[int]] = None