# Import the library.
# NOTE: Adjustments might be needed.
from mapper import ILPOptimizer, model


# Initial placement example.
opt = ILPOptimizer(
    current_placement={},
    used_host_dstores={},
    used_shared_dstores={},
    vm_requirements=[
        model.VMRequirements(
            id=0,
            state=model.VMState.PENDING,
            memory=1024,
            cpu_ratio=12.0,
            cpu_usage=2.0,
            host_ids={0}
        ),
        model.VMRequirements(
            id=1,
            state=model.VMState.PENDING,
            memory=512,
            cpu_ratio=6.0,
            cpu_usage=2.0,
            host_ids={0, 1}
        ),
    ],
    vm_groups=[],
    host_capacities=[
        model.HostCapacity(
            id=0,
            memory=model.Capacity(total=4096, usage=0),
            cpu=model.Capacity(total=32.0, usage=0),
            cluster_id=0,
            # Added by UmU.
            base_energy_consumption=1,
            per_core_energy_consumption=2,
            max_energy_consumption=100
        ),
        model.HostCapacity(
            id=1,
            memory=model.Capacity(total=4096, usage=0),
            cpu=model.Capacity(total=32.0, usage=0),
            cluster_id=0,
            # Added by UmU.
            base_energy_consumption=1,
            per_core_energy_consumption=2,
            max_energy_consumption=100
        )
    ],
    dstore_capacities=[],
    vnet_capacities=[],
    criteria='EIMP',
    allowed_migrations=None,
    balance_constraints=None,
    msg=False
)

opt.map()

print(opt.placements())


# Workload optimization example.
opt = ILPOptimizer(
    current_placement={0: 0, 1: 1},
    used_host_dstores={},
    used_shared_dstores={},
    vm_requirements=[
        model.VMRequirements(
            id=0,
            state=model.VMState.RUNNING,
            memory=1024,
            cpu_ratio=12.0,
            cpu_usage=2.0,
            host_ids={0}
        ),
        model.VMRequirements(
            id=1,
            state=model.VMState.RUNNING,
            memory=512,
            cpu_ratio=6.0,
            cpu_usage=2.0,
            host_ids={0, 1}
        ),
    ],
    vm_groups=[],
    host_capacities=[
        model.HostCapacity(
            id=0,
            memory=model.Capacity(total=4096, usage=1024),
            cpu=model.Capacity(total=32.0, usage=12.0),
            cluster_id=0,
            # Added by UmU.
            base_energy_consumption=20,
            per_core_energy_consumption=1,
            max_energy_consumption=100
        ),
        model.HostCapacity(
            id=1,
            memory=model.Capacity(total=4096, usage=512),
            cpu=model.Capacity(total=32.0, usage=6.0),
            cluster_id=0,
            # Added by UmU.
            base_energy_consumption=20,
            per_core_energy_consumption=1,
            max_energy_consumption=100
        )
    ],
    dstore_capacities=[],
    vnet_capacities=[],
    criteria='EIMP',
    allowed_migrations=None,
    balance_constraints=None,
    msg=False
)

opt.map()

print(opt.placements())

