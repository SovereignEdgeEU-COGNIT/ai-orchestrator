package opennebula

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/stretchr/testify/assert"
)

func setup() (map[string]*core.VM, error) {
	vmIDs, err := GetVMIDs(prometheusURL)

	if err != nil {
		return nil, err
	}

	vmMap := make(map[string]*core.VM)
	for _, vmID := range vmIDs {
		vm := &core.VM{
			VMID: vmID,
		}
		vmMap[vmID] = vm
	}
	return vmMap, nil
}

func TestGetVMIDs(t *testing.T) {
	vmIDs, err := GetVMIDs(prometheusURL)

	for _, vmID := range vmIDs {
		t.Log(vmID)
	}
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.NotEmpty(t, vmIDs, "vmIDs should not be empty")
}

func TestMapVMHostIDs(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = MapVMHostIDs(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	for _, vm := range vmMap {
		assert.NotEmpty(t, vm.HostID)
	}
}

func TestGetVMsDiskRead(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsDiskRead(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.DiskRead == 0 {
			invalidVMs[id] = vm.DiskRead
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 diskRead")
}

func TestGetVMsDiskWrite(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsDiskWrite(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.DiskWrite == 0 {
			invalidVMs[id] = vm.DiskWrite
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 diskWrite")
}

func TestGetVMsNetRx(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsNetRx(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.NetRX == 0 {
			invalidVMs[id] = vm.NetRX
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 netRX")
}

func TestGetVMsNetTx(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsNetTx(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.NetTX == 0 {
			invalidVMs[id] = vm.NetTX
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 netTX")
}

func TestGetVMsCPUUsage(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsCPUUsage(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.UsageCPU == 0 {
			invalidVMs[id] = vm.UsageCPU
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 CPU usage")
}

func TestGetVMsMemUsage(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsMemUsage(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.UsageMemory == 0 {
			invalidVMs[id] = vm.UsageMemory
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 memory usage")
}

func TestGetVMsCPUTotal(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsCPUTotal(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}

	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.TotalCPU == 0 {
			invalidVMs[id] = vm.TotalCPU
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 total CPU")
}

func TestGetVMsMemTotal(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsMemTotal(prometheusURL, vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	invalidVMs := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.TotalMemory == 0 {
			invalidVMs[id] = vm.TotalMemory
		}
	}
	assert.Empty(t, invalidVMs, "VMs with 0 total memory")
}
