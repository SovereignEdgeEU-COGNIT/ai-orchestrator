package server

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestAddVM(t *testing.T) {
	client, server, done := prepareTests(t)

	vm, err := client.GetVM("vm1")
	assert.Nil(t, err)
	assert.Nil(t, vm)

	vm = &core.VM{VMID: "vm1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddVM(vm)
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.NotNil(t, vm)
	assert.Equal(t, "vm1", vm.VMID)

	server.Shutdown()
	<-done
}

func TestGetVM(t *testing.T) {
	client, server, done := prepareTests(t)

	vm, err := client.GetVM("vm1")
	assert.Nil(t, err)
	assert.Nil(t, vm)

	vm = &core.VM{VMID: "vm1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddVM(vm)
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.NotNil(t, vm)
	assert.Equal(t, "vm1", vm.VMID)

	server.Shutdown()
	<-done
}

func TestGetVMs(t *testing.T) {
	client, server, done := prepareTests(t)

	vms, err := client.GetVMs()
	assert.Nil(t, err)
	assert.Nil(t, vms)

	vm := &core.VM{VMID: "vm1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddVM(vm)
	assert.Nil(t, err)

	vm = &core.VM{VMID: "vm2", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddVM(vm)
	assert.Nil(t, err)

	vms, err = client.GetVMs()
	assert.Nil(t, err)
	assert.NotNil(t, vms)
	assert.Len(t, vms, 2)

	server.Shutdown()
	<-done
}

func TestRemoveVM(t *testing.T) {
	client, server, done := prepareTests(t)

	vm := &core.VM{VMID: "vm1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err := client.AddVM(vm)
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.NotNil(t, vm)
	assert.Equal(t, "vm1", vm.VMID)

	err = client.RemoveVM("vm1")
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.Nil(t, vm)

	server.Shutdown()
	<-done
}

func TestBind(t *testing.T) {
	client, server, done := prepareTests(t)

	host := &core.Host{HostID: "host1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err := client.AddHost(host)
	assert.Nil(t, err)

	hostFromDB, err := client.GetHost("host1")
	assert.Nil(t, err)

	vm := &core.VM{VMID: "vm1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddVM(vm)
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.False(t, vm.Deployed)
	assert.Equal(t, "", vm.HostID)
	assert.Equal(t, 0, vm.HostStateID)

	err = client.Bind("vm1", "host1")
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.True(t, vm.Deployed)
	assert.Equal(t, hostFromDB.HostID, vm.HostID)
	assert.Equal(t, hostFromDB.StateID, vm.HostStateID)

	err = client.Unbind("vm1", "host1")
	assert.Nil(t, err)

	vm, err = client.GetVM("vm1")
	assert.Nil(t, err)
	assert.False(t, vm.Deployed)
	assert.Equal(t, "", vm.HostID)
	assert.Equal(t, 0, vm.HostStateID)

	server.Shutdown()
	<-done
}
