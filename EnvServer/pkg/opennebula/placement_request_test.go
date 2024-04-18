package opennebula

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParsePlacementRequest(t *testing.T) {
	jsonStr := `{
        "VMS": [
            {
                "CAPACITY": {
                    "CPU": 1.0,
                    "DISK_SIZE": 2252,
                    "MEMORY": 786432
                },
                "HOST_IDS": [
                    0,
                    2,
                    3,
                    4
                ],
                "ID": 7,
                "STATE": "PENDING",
                "USER_TEMPLATE": {
                    "LOGO": "images/logos/ubuntu.png",
                    "LXD_SECURITY_PRIVILEGED": "true",
                    "SCHED_REQUIREMENTS": "ID=\"0\" | ID=\"2\" | ID=\"3\" | ID=\"4\""
                }
            }
        ]
    }`

	placementRequest, err := ParsePlacementRequest(jsonStr)
	assert.Nil(t, err)

	placementRequest2 := PlacementRequest{
		VMs: []VM{
			{
				Capacity: Capacity{
					CPU:      1.0,
					DiskSize: 2252,
					Memory:   786432,
				},
				HostIDs: []int{0, 2, 3, 4},
				ID:      7,
				State:   "PENDING",
				UserTemplate: UserTemplate{
					Logo:                  "images/logos/ubuntu.png",
					LxdSecurityPrivileged: "true",
					SchedRequirements:     "ID=\"0\" | ID=\"2\" | ID=\"3\" | ID=\"4\"",
				},
			},
		},
	}

	assert.True(t, placementRequest.Equals(&placementRequest2))
	assert.False(t, placementRequest.Equals(nil))
}
