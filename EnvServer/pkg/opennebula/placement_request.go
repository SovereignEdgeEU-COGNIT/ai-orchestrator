package opennebula

import (
	"encoding/json"
)

type Capacity struct {
	CPU      float64 `json:"CPU"`
	DiskSize int     `json:"DISK_SIZE"`
	Memory   int     `json:"MEMORY"`
}

type UserTemplate struct {
	Logo                  string `json:"LOGO"`
	LxdSecurityPrivileged string `json:"LXD_SECURITY_PRIVILEGED"`
	SchedRequirements     string `json:"SCHED_REQUIREMENTS"`
}

type VM struct {
	Capacity     Capacity     `json:"CAPACITY"`
	HostIDs      []int        `json:"HOST_IDS"`
	ID           int          `json:"ID"`
	State        string       `json:"STATE"`
	UserTemplate UserTemplate `json:"USER_TEMPLATE"`
}

func (vm *VM) ToJSON() (string, error) {
	jsonBytes, err := json.Marshal(vm)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

type PlacementRequest struct {
	VMs []VM `json:"VMS"`
}

func (request *PlacementRequest) ToJSON() (string, error) {
	jsonBytes, err := json.Marshal(request)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (request *PlacementRequest) Equals(request2 *PlacementRequest) bool {
	if request2 == nil {
		return false
	}

	if len(request.VMs) != len(request2.VMs) {
		return false
	}

	for i := 0; i < len(request.VMs); i++ {
		if request.VMs[i].ID != request2.VMs[i].ID &&
			request.VMs[i].State != request2.VMs[i].State &&
			request.VMs[i].Capacity.CPU != request2.VMs[i].Capacity.CPU &&
			request.VMs[i].Capacity.DiskSize != request2.VMs[i].Capacity.DiskSize &&
			request.VMs[i].Capacity.Memory != request2.VMs[i].Capacity.Memory &&
			request.VMs[i].UserTemplate.Logo != request2.VMs[i].UserTemplate.Logo &&
			request.VMs[i].UserTemplate.LxdSecurityPrivileged != request2.VMs[i].UserTemplate.LxdSecurityPrivileged &&
			request.VMs[i].UserTemplate.SchedRequirements != request2.VMs[i].UserTemplate.SchedRequirements &&
			len(request.VMs[i].HostIDs) != len(request2.VMs[i].HostIDs) {
			return false
		}

		for j := 0; j < len(request.VMs[i].HostIDs); j++ {
			if request.VMs[i].HostIDs[j] != request2.VMs[i].HostIDs[j] {
				return false
			}
		}

		return true
	}

	return true
}

func ParsePlacementRequest(jsonString string) (*PlacementRequest, error) {
	var request *PlacementRequest
	//print(jsonString)
	err := json.Unmarshal([]byte(jsonString), &request)
	if err != nil {
		return nil, err
	}

	return request, nil
}
