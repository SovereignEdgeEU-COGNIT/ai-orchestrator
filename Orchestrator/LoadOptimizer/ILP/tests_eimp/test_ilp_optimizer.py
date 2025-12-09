# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import json
import os
from typing import Any, Union

from pulp import value

import pytest

from mapper.ilp_optimizer import ILPOptimizer
from mapper.model import (
    Capacity,
    DStoreCapacity,
    DStoreRequirement,
    HostCapacity,
    PCIDevice,
    PCIDeviceRequirement,
    VMGroup,
    VMRequirements,
    VMState,
    VNetCapacity
)


class TestILPOptimizer:
    # pylint: disable=protected-access, attribute-defined-outside-init
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data_path = os.path.join(
            os.getcwd(),
            # "spec",
            # "functionality",
            # "scheduler",
            # "one_drs",
            "tests_eimp",
            "data",
            "mapper"
        )

    def _load(self, path: Union[str, os.PathLike]) -> dict[str, Any]:
        file_path = os.path.join(self.data_path, path)
        with open(file_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)

        used_host_dstores = data.get('used_host_dstores') or {}
        used_shared_dstores: dict[tuple[int, int], dict[int, int]] = {}
        if 'used_shared_dstores' in data:
            used_shared_dstores = data['used_shared_dstores']
        else:
            used_shared_dstores = {}
            for pair in data.get('used_disks') or {}:
                used_shared_dstores[tuple(pair[0])] = pair[1]

        dstore_caps: list[DStoreCapacity] = []
        for dstore_cap_data in data.get('datastore_capacities') or []:
            dstore_cap = DStoreCapacity(
                id=int(dstore_cap_data['id']),
                size=Capacity(**dstore_cap_data['size']),
                cluster_ids=dstore_cap_data.get('cluster_ids') or [0]
            )
            dstore_caps.append(dstore_cap)

        dstore_ids = [cap.id for cap in dstore_caps]
        alloc: dict[int, int] = {}
        vm_reqs: list[VMRequirements] = []
        for vm_req_data in data['vm_requirements']:
            if (host_id := vm_req_data.get('host_id')) is not None:
                alloc[vm_req_data['id']] = host_id
            vm_id = int(vm_req_data['id'])
            storage: dict[int, DStoreRequirement] = {}
            if 'storage' in vm_req_data:
                for id_, storage_data in vm_req_data['storage'].items():
                    storage[id_] = DStoreRequirement(
                        id=id_, vm_id=vm_id, **storage_data
                    )
            else:
                storage = {
                    0: DStoreRequirement(
                        id=0,
                        vm_id=vm_id,
                        size=int(vm_req_data['disk_size']),
                        allow_host_dstores=True,
                        shared_dstore_ids=dstore_ids,
                    )
                }
            vm_req = VMRequirements(
                id=vm_id,
                state=VMState(vm_req_data['state']),
                memory=int(vm_req_data['memory']),
                cpu_ratio=float(vm_req_data['cpu_ratio']),
                cpu_usage=float(vm_req_data['cpu_usage']),
                # HACK: This is a temporary solution.
                storage=storage,
                disk_usage=float(vm_req_data.get('disk_usage', 'nan')),
                pci_devices=[
                    PCIDeviceRequirement(**pcid_req_data)
                    for pcid_req_data in vm_req_data.get('pci_devices') or {}
                ],
                host_ids=set(vm_req_data['host_ids']),
                # HACK: This is a temporary solution because
                # `vm_req_data.get('nic_matches')` is a list instead of
                # a dict.
                # TODO: Modify JSON files to represent NIC matches as
                # dicts with NIC IDs as keys.
                share_vnets=vm_req_data.get('share_vnets', True),
                nic_matches=dict(
                    enumerate(vm_req_data.get('nic_matches') or [])
                ),
                net_usage=float(vm_req_data.get('net_usage', 'nan'))
            )
            vm_reqs.append(vm_req)

        vm_groups: list[VMGroup] = []
        for vm_group_data in data.get('vm_groups') or []:
            vm_group = VMGroup(
                id=int(vm_group_data['id']),
                affined=bool(vm_group_data['affined']),
                vm_ids=set(vm_group_data['vm_ids'])
            )
            vm_groups.append(vm_group)

        host_caps: list[HostCapacity] = []
        for host_cap_data in data['host_capacities']:
            memory_data = host_cap_data['memory']
            cpu_data = host_cap_data['cpu']
            disks: dict[int, Capacity] = {}
            if 'disks' in host_cap_data:
                for id_, disk_data in host_cap_data['disks'].items():
                    disks[id_] = Capacity(**disk_data)
            elif 'disk' in host_cap_data:
                disks = {0: Capacity(**host_cap_data['disk'])}
            else:
                disks = {0: Capacity(0, 0)}
            host_cap = HostCapacity(
                id=int(host_cap_data['id']),
                memory=Capacity(**memory_data),
                cpu=Capacity(**cpu_data),
                base_energy_consumption=float(host_cap_data.get('base_energy_consumption', 100.0)),
                per_core_energy_consumption=float(host_cap_data.get('per_core_energy_consumption', 10.0)),
                max_energy_consumption=float(host_cap_data.get('max_energy_consumption', 1000.0)),
                disks=disks,
                disk_io=Capacity(host_cap_data.get('disk_io', 0.0), 0.0),
                net=Capacity(host_cap_data.get('net', 0.0), 0.0),
                pci_devices=[
                    PCIDevice(**pcid_data)
                    for pcid_data in host_cap_data.get('pci_devices') or {}
                ],
                cluster_id=host_cap_data.get('cluster_id', 0)
            )
            host_caps.append(host_cap)

        vnet_caps: list[VNetCapacity] = []
        for vnet_cap_data in data.get('vnet_capacities') or []:
            vnet_cap = VNetCapacity(
                id=int(vnet_cap_data['id']),
                n_free_ip_addresses=int(vnet_cap_data['n_free_ip_addresses']),
                cluster_ids=vnet_cap_data.get('cluster_ids') or [0]
            )
            vnet_caps.append(vnet_cap)

        out = {
            'current_placement': alloc,
            'used_host_dstores': used_host_dstores,
            'used_shared_dstores': used_shared_dstores,
            'vm_requirements': vm_reqs,
            'vm_groups': vm_groups,
            'host_capacities': host_caps,
            'dstore_capacities': dstore_caps,
            'vnet_capacities': vnet_caps,
            # 'verbose': 0,
            # 'solver': 'COIN_CMD',
            'msg': False
        }
        return out

    def _placement(self, optimizer: ILPOptimizer) -> dict[int, int]:
        out: dict[int, int] = {}
        for vm_id, alloc in optimizer._opt_placement.items():
            if alloc is not None:
                out[vm_id] = alloc.host_id
        return out

    def test_init_workload_balance(self):
        data = self._load('data_00.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        assert opt._curr_alloc == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
        assert not opt._narrow
        vm_host_matches = {
            vm_id: sorted(host_cap.id for host_cap in host_caps)
            for vm_id, host_caps in opt._vm_host_matches.items()
        }
        assert vm_host_matches == {
            0: [0],
            1: [0],
            2: [1],
            3: [0, 1, 2],
            4: [0, 1, 2],
            5: [0, 1],
            6: [0],
            7: [0, 1, 2],
            8: [0, 1, 2]
        }
        host_vm_matches = {
            host_id: sorted(vm_req.id for vm_req in vm_reqs)
            for host_id, vm_reqs in opt._host_vm_matches.items()
        }
        assert host_vm_matches == {
            0: [0, 1, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 5, 7, 8], 2: [3, 4, 7, 8]
        }
        assert len(opt._pcid_matches) == 8

    def test_init_maintenance_narrow(self):
        data = self._load('data_01.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        assert opt._curr_alloc == {0: 0, 1: 0, 2: 0}
        assert opt._narrow
        vm_host_matches = {
            vm_id: sorted(host_cap.id for host_cap in host_caps)
            for vm_id, host_caps in opt._vm_host_matches.items()
        }
        assert vm_host_matches == {0: [1], 1: [1, 2], 2: [1, 2]}
        host_vm_matches = {
            host_id: sorted(vm_req.id for vm_req in vm_reqs)
            for host_id, vm_reqs in opt._host_vm_matches.items()
        }
        assert host_vm_matches == {1: [0, 1, 2], 2: [1, 2]}

    def test_init_maintenance_full(self):
        data = self._load('data_03.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        assert opt._curr_alloc == {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
        assert not opt._narrow
        vm_host_matches = {
            vm_id: sorted(host_cap.id for host_cap in host_caps)
            for vm_id, host_caps in opt._vm_host_matches.items()
        }
        assert vm_host_matches == {
            0: [1], 1: [1, 2], 2: [1, 2], 3: [1], 4: [1, 2]
        }
        host_vm_matches = {
            host_id: sorted(vm_req.id for vm_req in vm_reqs)
            for host_id, vm_reqs in opt._host_vm_matches.items()
        }
        assert host_vm_matches == {1: [0, 1, 2, 3, 4], 2: [1, 2, 4]}

    def test_init_placement_narrow(self):
        data = self._load('data_05.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        assert not opt._curr_alloc
        assert opt._narrow
        vm_host_matches = {
            vm_id: sorted(host_cap.id for host_cap in host_caps)
            for vm_id, host_caps in opt._vm_host_matches.items()
        }
        assert vm_host_matches == {0: [1], 1: [1, 2], 2: [1, 2]}
        host_vm_matches = {
            host_id: sorted(vm_req.id for vm_req in vm_reqs)
            for host_id, vm_reqs in opt._host_vm_matches.items()
        }
        assert host_vm_matches == {1: [0, 1, 2], 2: [1, 2]}

    def test_init_placement_full(self):
        data = self._load('data_07.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        assert opt._curr_alloc == {3: 1, 4: 2}
        assert not opt._narrow
        vm_host_matches = {
            vm_id: sorted(host_cap.id for host_cap in host_caps)
            for vm_id, host_caps in opt._vm_host_matches.items()
        }
        assert vm_host_matches == {
            0: [1], 1: [1, 2], 2: [1, 2], 3: [1], 4: [1, 2]
        }
        host_vm_matches = {
            host_id: sorted(vm_req.id for vm_req in vm_reqs)
            for host_id, vm_reqs in opt._host_vm_matches.items()
        }
        assert host_vm_matches == {1: [0, 1, 2, 3, 4], 2: [1, 2, 4]}

    def test_map_workload_balance_memory_balance(self):
        data = self._load('data_00.json')
        opt = ILPOptimizer(**data, criteria='memory_balance')
        opt.map()
        result = {0: 0, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 0, 7: 2, 8: 1}
        assert self._placement(opt) == result

    def test_map_workload_balance_cpu_usage_balance(self):
        data = self._load('data_00.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        host_ids = list(self._placement(opt).values())
        n_host_ids = {host_id: host_ids.count(host_id) for host_id in host_ids}
        assert n_host_ids == {0: 3, 1: 4, 2: 2}

    def test_map_workload_balance_pack(self):
        data = self._load('data_00.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        # Host with the ID 2 is not used.
        assert 2 not in self._placement(opt).values()

    def test_map_maintenance_cpu_usage_balance_narrow(self):
        data = self._load('data_01.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 2, 2: 1}

    def test_map_maintenance_cpu_usage_balance_narrow_infeas(self):
        data = self._load('data_02.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert not opt._vm_host_matches[1]
        assert 1 not in self._placement(opt)

    def test_map_maintenance_pack_narrow(self):
        data = self._load('data_01.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 1, 2: 1}

    def test_map_maintenance_pack_full(self):
        data = self._load('data_03.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

    def test_map_maintenance_cpu_usage_balance_infeas(self):
        data = self._load('data_04.json')
        del data['host_capacities'][0]
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert opt._model.status == -1  # Infeasible.

    def test_map_placement_narrow_cpu_usage_balance(self):
        data = self._load('data_05.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 2, 2: 1}

    def test_map_placement_narrow_cpu_usage_balance_infeas(self):
        data = self._load('data_06.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        alloc = self._placement(opt)
        assert not opt._vm_host_matches[1]
        assert 1 not in alloc
        assert alloc == {0: 1, 2: 2}

    def test_map_placement_narrow_pack(self):
        data = self._load('data_05.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 1, 2: 1}

    def test_map_placement_full_pack(self):
        data = self._load('data_07.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()
        assert self._placement(opt) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

    def test_map_placement_full_pack_infeas(self):
        data = self._load('data_08.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert len(opt._pcid_matches) == 3
        assert self._placement(opt) == {0: 1, 2: 1, 3: 1, 4: 1}

    def test_map_placement_narrow_infeas_cpu_shortage(self):
        data = self._load('data_09.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()
        assert len(opt._vm_host_matches[0]) == 1
        assert len(opt._vm_host_matches[1]) == 1
        assert len(opt._host_vm_matches[0]) == 2
        assert len(self._placement(opt)) == 1

    def test_map_placement_narrow_infeas_memory_shortage(self):
        data = self._load('data_10.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()
        assert len(opt._vm_host_matches[0]) == 1
        assert len(opt._vm_host_matches[1]) == 1
        assert len(opt._host_vm_matches[0]) == 2
        assert len(self._placement(opt)) == 1

    def test_map_placement_narrow_infeas(self):
        data = self._load('data_11.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        assert not self._placement(opt)

    def test_map_vnets(self):
        data = self._load('data_12.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        vnets = {
            idx for idx, x_var in opt._x_vnet.items() if round(value(x_var))
        }

        assert set(opt._x_vnet) == {
            (5, 0, 10), (5, 0, 11), (5, 1, 11), (5, 2, 12), (6, 0, 10)
        }
        assert vnets == {(5, 0, 11), (5, 1, 11), (5, 2, 12), (6, 0, 10)}
        assert set(self._placement(opt)) == {0, 5, 6}

    def test_map_vnets_shortage(self):
        data = self._load('data_13.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()
        alloc = self._placement(opt)

        assert set(opt._x_vnet) == {
            (5, 0, 10), (5, 0, 11), (5, 1, 11), (5, 2, 12), (6, 0, 10)
        }
        assert len(alloc) == 2
        assert 5 not in alloc or 6 not in alloc

    def test_map_anti_affinity(self):
        data = self._load('data_14.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        alloc = self._placement(opt)
        assert len(alloc) == 6
        assert alloc[5] != alloc[6]
        assert alloc[6] != alloc[7]
        assert alloc[5] != alloc[7]
        assert alloc[8] != alloc[9]

    def test_map_anti_affinity_conflict(self):
        data = self._load('data_15.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        result = self._placement(opt)
        assert len(result) == 5
        assert 8 not in result or 9 not in result

    def test_map_affinity_mixed(self):
        data = self._load('data_16.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1}

    def test_map_affinity_mixed_with_migrations(self):
        data = self._load('data_17.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1}

    def test_map_affinity_mixed_infeasible(self):
        data = self._load('data_18.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 1, 7: 1, 8: 1, 9: 1, 10: 0}

    def test_map_affinity_placement_narrow(self):
        data = self._load('data_19.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0}

    def test_map_affinity_placement_infeasible(self):
        data = self._load('data_20.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()

        assert self._placement(opt) == {}

    def test_map_affinity_resched(self):
        data = self._load('data_21.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()

        assert self._placement(opt) == {8: 0, 9: 0, 10: 0}

    def test_map_datastore_mixed(self):
        data = self._load('data_22.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_map_datastore_mixed_infeas(self):
        data = self._load('data_23.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {6: 0, 8: 0, 9: 0}

    def test_map_datastore_placement_local_storage(self):
        data = self._load('data_24.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {5: 1, 6: 0}

    def test_map_datastore_placement_cluster_storage_narrow(self):
        data = self._load('data_25.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0}

    def test_map_datastore_affinity_placement_cluster_storage_narrow(self):
        data = self._load('data_26.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 0}

    def test_map_datastore_affinity_placement_cluster_storage_infeas(self):
        data = self._load('data_27.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()

        assert self._placement(opt) == {}

    def test_map_datastore_maintenance(self):
        data = self._load('data_28.json')
        opt = ILPOptimizer(**data, criteria='EIMP')
        opt.map()

        assert self._placement(opt) in [
            {8: 0, 9: 0, 10: 0}, {8: 1, 9: 1, 10: 1}
        ]

    def test_map_cluster_datastore_placement_pack(self):
        data = self._load('data_29.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) in [{5: 0, 6: 0, 7: 2}, {5: 2, 6: 0, 7: 2}]

    def test_map_cluster_datastore_placement_balance(self):
        data = self._load('data_29.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()

        assert self._placement(opt) == {5: 1, 6: 0, 7: 2}

    def test_map_cluster_datastore_placement_pack_infeas(self):
        data = self._load('data_30.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()
        alloc = self._placement(opt)

        assert len(alloc) == 2
        assert len(set(alloc.values())) == 1

    def test_map_cluster_datastore_placement_balance_infeas(self):
        data = self._load('data_30.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        alloc = self._placement(opt)

        assert len(alloc) == 2
        assert len(set(alloc.values())) == 2

    def test_map_cluster_vnet_placement_pack(self):
        data = self._load('data_31.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 0, 7: 2}

    def test_map_cluster_vnet_placement_balance(self):
        data = self._load('data_31.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()

        assert self._placement(opt) == {5: 1, 6: 0, 7: 2}

    def test_map_cluster_vnet_placement_pack_infeas(self):
        data = self._load('data_32.json')
        opt = ILPOptimizer(**data, criteria='pack')
        opt.map()

        assert self._placement(opt) == {5: 2, 7: 2}

    def test_map_cluster_vnet_placement_balance_infeas(self):
        data = self._load('data_32.json')
        opt = ILPOptimizer(**data, criteria='cpu_usage_balance')
        opt.map()
        alloc = self._placement(opt)

        assert alloc[5] in {0, 1}
        assert alloc[7] == 2

    def test_map_placement_narrow_objectives(self):
        data = self._load('data_33.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 1, 7: 2}

        opt = ILPOptimizer(**data, criteria='memory_balance')
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 1, 7: 2}

        opt = ILPOptimizer(
            **data, criteria={'cpu_ratio_balance': 0.5, 'memory_balance': 0.5}
        )
        opt.map()

        assert self._placement(opt) == {5: 0, 6: 1, 7: 2}

    def test_map_resched_objectives(self):
        data = self._load('data_34.json')
        opt = ILPOptimizer(**data, criteria='cpu_ratio_balance')
        opt.map()

        assert set(self._placement(opt).values()) == {0, 2}
        assert opt._balance['cpu_ratio'].value() == 0.5

        opt = ILPOptimizer(**data, criteria='memory_balance')
        opt.map()

        assert set(self._placement(opt).values()) == {0, 3}
        assert opt._balance['memory'].value() == 0.5

        opt = ILPOptimizer(
            **data, criteria={'cpu_ratio_balance': 0.5, 'memory_balance': 0.5}
        )
        opt.map()

        assert set(self._placement(opt).values()) == {0, 1}
        assert round(opt._balance['cpu_ratio'].value(), 2) == round(4 / 7, 2)
        assert round(opt._balance['memory'].value(), 2) == round(2 / 3, 2)

        opt = ILPOptimizer(**data, criteria='disk_usage_balance')
        opt.map()

        assert set(self._placement(opt).values()) == {0, 2}
        assert opt._balance['disk_usage'].value() == 0.5

        opt = ILPOptimizer(**data, criteria='net_usage_balance')
        opt.map()

        assert set(self._placement(opt).values()) == {0, 3}
        assert opt._balance['net_usage'].value() == 0.5

        opt = ILPOptimizer(
            **data,
            criteria={'disk_usage_balance': 0.5, 'net_usage_balance': 0.5}
        )
        opt.map()

        assert set(self._placement(opt).values()) == {0, 1}
        assert round(opt._balance['disk_usage'].value(), 2) == round(4 / 7, 2)
        assert round(opt._balance['net_usage'].value(), 2) == round(2 / 3, 2)
