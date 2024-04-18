package core

import "encoding/json"

type VM struct {
	VMID        string  `json:"vmid"`
	StateID     int     `json:"stateid"`
	Deployed    bool    `json:"deployed"`
	HostID      string  `json:"hostid"`
	HostStateID int     `json:"hoststateid"`
	TotalCPU    float64 `json:"total_cpu"`
	TotalMemory float64 `json:"total_memory"`
	UsageCPU    float64 `json:"usage_cpu"`
	UsageMemory float64 `json:"usage_memory"`
	DiskRead    float64 `json:"disk_read"`
	DiskWrite   float64 `json:"disk_write"`
	NetRX       float64 `json:"netrx"`
	NetTX       float64 `json:"nettx"`
}

func ConvertJSONToVM(jsonString string) (*VM, error) {
	var vm *VM
	err := json.Unmarshal([]byte(jsonString), &vm)
	if err != nil {
		return nil, err
	}

	return vm, nil
}

func ConvertJSONToVMArray(jsonString string) ([]*VM, error) {
	var vms []*VM

	err := json.Unmarshal([]byte(jsonString), &vms)
	if err != nil {
		return vms, err
	}

	return vms, nil
}

func ConvertVMArrayToJSON(vms []*VM) (string, error) {
	jsonBytes, err := json.Marshal(vms)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (vm *VM) ToJSON() (string, error) {
	jsonBytes, err := json.Marshal(vm)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (vm *VM) Equals(vm2 *VM) bool {
	if vm2 == nil {
		return false
	}

	if vm.VMID == vm2.VMID &&
		vm.StateID == vm2.StateID &&
		vm.Deployed == vm2.Deployed &&
		vm.HostID == vm2.HostID &&
		vm.HostStateID == vm2.HostStateID &&
		vm.TotalCPU == vm2.TotalCPU &&
		vm.TotalMemory == vm2.TotalMemory &&
		vm.UsageCPU == vm2.UsageCPU &&
		vm.UsageMemory == vm2.UsageMemory &&
		vm.DiskRead == vm2.DiskRead &&
		vm.DiskWrite == vm2.DiskWrite &&
		vm.NetRX == vm2.NetRX &&
		vm.NetTX == vm2.NetTX {
		return true
	}

	return false
}

func IsVMArraysEqual(vms []*VM, vms2 []*VM) bool {
	if len(vms) != len(vms2) {
		return false
	}

	for i := 0; i < len(vms); i++ {
		if !vms[i].Equals(vms2[i]) {
			return false
		}
	}

	return true
}
