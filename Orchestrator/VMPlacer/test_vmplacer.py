#!/usr/bin/env python3

from typing import Dict, Optional
from bestFitMapper import BestFitMapper
from model import VMRequirements, HostCapacity, Capacity, VMState, Allocation

def print_test_case(name: str, vms: list[VMRequirements], hosts: list[HostCapacity], result: Dict[int, Optional[Allocation]] = None, error: Exception = None):
    """Helper function to print test case details in a readable format"""
    print(f"\n{'='*80}")
    if error:
        print(f"Test Case: {name} - FAILED")
        print('='*80)
    else:
        print(f"Test Case: {name}")
        print('='*80)
    
    print("\nInput VMs:")
    for vm in vms:
        print(f"VM {vm.id}: Memory={vm.memory}GB, CPU={vm.cpu_ratio:.2f}, State={vm.state}, "
              f"Allowed Hosts={vm.host_ids if vm.host_ids else 'Not specified'}")
    
    print("\nAvailable Hosts:")
    for host in hosts:
        print(f"Host {host.id}: Memory={host.memory.total}GB (Used: {host.memory.usage}GB), "
              f"CPU={host.cpu.total:.2f} (Used: {host.cpu.usage:.2f})")
    
    if error:
        print(f"\nError Type: {error.__class__.__name__}")
        print(f"Error Message: {str(error)}")
    elif result is not None:
        print("\nAllocation Results:")
        successful_allocations = 0
        failed_allocations = 0
        for vm_id, allocation in result.items():
            if allocation:
                print(f"VM {vm_id} → Host {allocation.host_id}")
                successful_allocations += 1
            else:
                print(f"VM {vm_id} → No suitable host found")
                failed_allocations += 1
        
        print(f"\nSummary:")
        print(f"- Total VMs: {len(result)}")
        print(f"- Successfully allocated: {successful_allocations}")
        print(f"- Failed to allocate: {failed_allocations}")
    print("\n")

def run_test_case(name: str, vms: list[VMRequirements], hosts: list[HostCapacity]) -> None:
    """Helper function to run a test case and handle potential errors"""
    try:
        mapper = BestFitMapper(current_placement={}, vm_requirements=vms,
                             host_capacities=hosts, criteria=None)
        mapper.map()
        result = mapper.placements(top_k=1)[0]
        print_test_case(name, vms, hosts, result)
    except (ValueError, KeyError) as e:
        print_test_case(name, vms, hosts, error=e)
    except Exception as e:
        # Handle unexpected exceptions differently
        print_test_case(name, vms, hosts, error=e)
        print(f"Unexpected error type: {type(e).__name__}")
        raise  # Re-raise unexpected exceptions

def main():
    print("Running VM Optimizer test cases...")
    
    # Test Case 1: Basic Allocation
    vms_basic = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={1, 2}
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=4, memory_usage=0,
            cpu_ratio=4.0, cpu_usage=0.0, host_ids={1, 2}
        )
    ]
    
    hosts_basic = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        ),
        HostCapacity(
            id=2,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        )
    ]
    
    run_test_case("Basic Allocation", vms_basic, hosts_basic)
    
    # Test Case 2: Resource Intensive VMs
    vms_intensive = [
        # CPU-intensive VM
        VMRequirements(
            id=1, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=6.0, cpu_usage=0.0, host_ids={1, 2}
        ),
        # Memory-intensive VM
        VMRequirements(
            id=2, state=VMState.PENDING, memory=6, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={1, 2}
        )
    ]
    
    hosts_intensive = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        ),
        HostCapacity(
            id=2,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        )
    ]
    
    run_test_case("Resource Intensive VMs", vms_intensive, hosts_intensive)
    
    # Test Case 3: Edge Cases - Full Host Capacity
    vms_full = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=7, memory_usage=0,
            cpu_ratio=7.0, cpu_usage=0.0, host_ids={1}
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=7, memory_usage=0,
            cpu_ratio=7.0, cpu_usage=0.0, host_ids={1}
        )
    ]
    
    hosts_full = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=7),
            cpu=Capacity(total=8.0, usage=7.0)
        )
    ]
    
    run_test_case("Full Host Capacity", vms_full, hosts_full)
    
    # Test Case 4: VMs with Specific Host Requirements
    vms_specific = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={1}
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={2}
        ),
        VMRequirements(
            id=3, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={1, 2}
        )
    ]
    
    hosts_specific = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        ),
        HostCapacity(
            id=2,
            memory=Capacity(total=8, usage=4),
            cpu=Capacity(total=8.0, usage=4.0)
        )
    ]
    
    run_test_case("Specific Host Requirements", vms_specific, hosts_specific)
    
    # Test Case 5: Error Case - Missing Host IDs
    vms_error = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0  # Intentionally missing host_ids
        )
    ]
    
    hosts_error = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        )
    ]
    
    run_test_case("Missing Host IDs", vms_error, hosts_error)

    # Test Case 6: Large Scale Allocation
    vms_large = [
        VMRequirements(
            id=i, state=VMState.PENDING, 
            memory=2 if i % 2 == 0 else 4,  # Alternating memory requirements
            memory_usage=0,
            cpu_ratio=2.0 if i % 2 == 0 else 4.0,  # Alternating CPU requirements
            cpu_usage=0.0,
            host_ids={1, 2, 3, 4, 5}
        ) for i in range(1, 11)  # 10 VMs
    ]
    
    hosts_large = [
        HostCapacity(
            id=i,
            memory=Capacity(total=16, usage=0),
            cpu=Capacity(total=16.0, usage=0.0)
        ) for i in range(1, 6)  # 5 hosts
    ]
    
    run_test_case("Large Scale Allocation", vms_large, hosts_large)

    # Test Case 7: Invalid Host IDs
    vms_invalid_hosts = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={99, 100}  # Non-existent hosts
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=2, memory_usage=0,
            cpu_ratio=2.0, cpu_usage=0.0, host_ids={1, 99}  # Mix of valid and invalid hosts
        )
    ]
    
    hosts_invalid = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        )
    ]
    
    run_test_case("Invalid Host IDs", vms_invalid_hosts, hosts_invalid)

    # Test Case 8: Heterogeneous Hosts
    vms_hetero = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=4, memory_usage=0,
            cpu_ratio=4.0, cpu_usage=0.0, host_ids={1, 2, 3}
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=8, memory_usage=0,
            cpu_ratio=8.0, cpu_usage=0.0, host_ids={1, 2, 3}
        ),
        VMRequirements(
            id=3, state=VMState.PENDING, memory=16, memory_usage=0,
            cpu_ratio=16.0, cpu_usage=0.0, host_ids={1, 2, 3}
        )
    ]
    
    hosts_hetero = [
        HostCapacity(  # Small host
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        ),
        HostCapacity(  # Medium host
            id=2,
            memory=Capacity(total=16, usage=0),
            cpu=Capacity(total=16.0, usage=0.0)
        ),
        HostCapacity(  # Large host
            id=3,
            memory=Capacity(total=32, usage=0),
            cpu=Capacity(total=32.0, usage=0.0)
        )
    ]
    
    run_test_case("Heterogeneous Hosts", vms_hetero, hosts_hetero)

    # Test Case 9: Resource Threshold Edge Cases
    vms_threshold = [
        VMRequirements(
            id=1, state=VMState.PENDING, memory=6, memory_usage=0,  # Just under 80% of 8GB
            cpu_ratio=6.0, cpu_usage=0.0, host_ids={1}  # Just under 80% of 8 CPU
        ),
        VMRequirements(
            id=2, state=VMState.PENDING, memory=7, memory_usage=0,  # Over 80% of 8GB
            cpu_ratio=7.0, cpu_usage=0.0, host_ids={1}  # Over 80% of 8 CPU
        )
    ]
    
    hosts_threshold = [
        HostCapacity(
            id=1,
            memory=Capacity(total=8, usage=0),
            cpu=Capacity(total=8.0, usage=0.0)
        )
    ]
    
    run_test_case("Resource Threshold Edge Cases", vms_threshold, hosts_threshold)

if __name__ == "__main__":
    main() 