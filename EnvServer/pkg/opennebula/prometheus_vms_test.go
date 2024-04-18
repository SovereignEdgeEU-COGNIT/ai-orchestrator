package opennebula

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/stretchr/testify/assert"
)

func setup() (map[string]*core.VM, error) {
	vmIDs, err := GetVMIDs(getPrometheusURL())

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
	vmIDs, err := GetVMIDs(getPrometheusURL())

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

	err = MapVMHostIDs(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	vmMappedToHost := false
	for _, vm := range vmMap {

		if vm.HostID != "" {
			vmMappedToHost = true
			break
		}
	}
	assert.True(t, vmMappedToHost, "No VMs mapped to host")
}

func TestGetVMsDiskRead(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsDiskRead(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.DiskRead != 0 {
			validVms[id] = vm.DiskRead
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 diskRead")
}

func TestGetVMsDiskWrite(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsDiskWrite(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.DiskWrite != 0 {
			validVms[id] = vm.DiskWrite
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 diskWrite")
}

func TestGetVMsNetRx(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsNetRx(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.NetRX != 0 {
			validVms[id] = vm.NetRX
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 netRX")
}

func TestGetVMsNetTx(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsNetTx(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.NetTX != 0 {
			validVms[id] = vm.NetTX
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 netTX")
}

func TestGetVMsCPUUsage(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsCPUUsage(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.UsageCPU != 0 {
			validVms[id] = vm.UsageCPU
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 CPU usage")
}

func TestGetVMsMemUsage(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsMemUsage(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.UsageMemory != 0 {
			validVms[id] = vm.UsageMemory
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 memory usage")
}

func TestGetVMsCPUTotal(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsCPUTotal(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}

	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.TotalCPU != 0 {
			validVms[id] = vm.TotalCPU
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 total CPU")
}

func TestGetVMsMemTotal(t *testing.T) {
	vmMap, err := setup()
	assert.Nil(t, err)

	err = GetVMsMemTotal(getPrometheusURL(), vmMap)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	validVms := make(map[string]float64)

	for id, vm := range vmMap {
		if vm.TotalMemory != 0 {
			validVms[id] = vm.TotalMemory
		}
	}
	assert.NotEmpty(t, validVms, "All VMs have 0 total memory")
}
