package opennebula

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParsePlacementResponse(t *testing.T) {
	jsonStr := `{
        "VMS": [
            {
                "ID": 7,
                "HOST_ID": 4
            }
        ]
    }`

	response, err := ParsePlacementResponse(jsonStr)
	assert.Nil(t, err)

	response2 := PlacementResponse{
		VMS: []VMMapping{
			{
				ID:     7,
				HostID: 4,
			},
		},
	}

	assert.True(t, response.Equals(&response2))
	assert.False(t, response.Equals(nil))
}
