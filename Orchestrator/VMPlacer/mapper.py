import abc
from collections.abc import Collection, Mapping
from typing import Any, Optional

from model import Allocation, HostCapacity, VMRequirements

class Mapper(abc.ABC):
    """
    Abstract base class for implementing Solvers for the optimal allocation.
    This class defines the interface for mapping VMs to hosts based on
    requirements and capacity constraints.

    Parameters
    ----------
    current_placement : Mapping[int, int]
        Current allocation of VMs to Hosts.
    vm_requirements : Collection[VMRequirements]
        Collection of VM requirements including memory and CPU needs.
    host_capacities : Collection[HostCapacity]
        Collection of host capacities including available resources.
    criteria : Any
        Mapping criteria to be used (e.g. packing, load_balance, ...)
    **kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    map()
        Solves the allocation.
    placements(top_k=1)
        Returns the top K placement solutions.
    """
    
    __slots__ = ()

    @abc.abstractmethod
    def __init__(
        self,
        current_placement: Mapping[int, int],
        vm_requirements: Collection[VMRequirements],
        host_capacities: Collection[HostCapacity],
        criteria: Any,
        **kwargs
    ) -> None:
        """
        Initialize the mapper with the model.

        Parameters
        ----------
        current_placement : Mapping[int, int]
            Current allocation mapping of VMs to Hosts.
        vm_requirements : Collection[VMRequirements]
            Collection of VM requirements including memory and CPU needs.
        host_capacities : Collection[HostCapacity]
            Collection of host capacities including available resources.
        criteria : Any
            Mapping criteria to be used.
        **kwargs : dict
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            When called without implementation in child class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def map(self) -> None:
        """
        Solved the mapping operation.

        This method should implement the logic for mapping VMs to hosts
        based on the initialized requirements and constraints.

        Raises
        ------
        NotImplementedError
            When called without implementation in child class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
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

        Raises
        ------
        NotImplementedError
            When called without implementation in child class.
        """
        raise NotImplementedError()