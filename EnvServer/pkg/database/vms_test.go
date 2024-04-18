package database

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestAddVM(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	vm, err := db.GetVM("test_vm_id")
	assert.Nil(t, err)
	assert.Nil(t, vm)

	vm = &core.VM{VMID: "test_vm_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vms, err := db.GetVMs()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(vms))

	vm, err = db.GetVM("test_vm_id")
	assert.Nil(t, err)
	assert.NotNil(t, vm)
	assert.Equal(t, "test_vm_id", vm.VMID)
}

func TestVMStateMetric(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	vm := &core.VM{VMID: "test_vm1_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vm = &core.VM{VMID: "test_vm2_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vms, err := db.GetVMs()
	assert.Nil(t, err)

	counter := 0
	for _, vm := range vms {
		counter += vm.StateID
	}
	assert.Equal(t, 3, counter)

	// If we remove a vm and add another one, the state ID should be reused
	err = db.RemoveVM("test_vm1_id")
	assert.Nil(t, err)

	vm = &core.VM{VMID: "test_vm3_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vms, err = db.GetVMs()
	assert.Nil(t, err)

	counter = 0
	for _, vm := range vms {
		counter += vm.StateID
	}
	assert.Equal(t, 3, counter)
}

func TestSetVMResources(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	vm := &core.VM{VMID: "test_vm_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	err = db.SetVMResources(vm.VMID, float64(1), float64(2), float64(3), float64(4), float64(5), float64(6))
	assert.Nil(t, err)

	hosts, err := db.GetVMs()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(hosts))
	assert.Equal(t, float64(1), hosts[0].UsageCPU)
	assert.Equal(t, float64(2), hosts[0].UsageMemory)
}

func TestRemoveVM(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	vm := &core.VM{VMID: "test_vm_id", Deployed: true, HostID: "test_host_id", HostStateID: 0, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vms, err := db.GetVMs()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(vms))

	err = db.RemoveVM(vm.VMID)
	assert.Nil(t, err)

	vms, err = db.GetVMs()
	assert.Nil(t, err)
	assert.Equal(t, 0, len(vms))
}

func TestBind(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	host := &core.Host{HostID: "test_host_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddHost(host)
	assert.Nil(t, err)

	vm := &core.VM{VMID: "test_vm_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = db.AddVM(vm)
	assert.Nil(t, err)

	vm, err = db.GetVM("test_vm_id")
	assert.Nil(t, err)
	assert.False(t, vm.Deployed)

	err = db.Bind(vm.VMID, host.HostID)
	assert.Nil(t, err)

	vm, err = db.GetVM("test_vm_id")
	assert.Nil(t, err)
	assert.True(t, vm.Deployed)
	assert.Equal(t, vm.HostID, vm.HostID)
	assert.Equal(t, vm.StateID, vm.HostStateID)

	err = db.Unbind(vm.VMID)
	assert.Nil(t, err)

	vm, err = db.GetVM("test_vm_id")
	assert.Nil(t, err)
	assert.Equal(t, "", vm.HostID)
	assert.Equal(t, 0, vm.HostStateID)
	assert.False(t, vm.Deployed)
}
