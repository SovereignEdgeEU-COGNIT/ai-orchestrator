package opennebula

import "encoding/json"

type VMMapping struct {
	ID     int `json:"ID"`
	HostID int `json:"HOST_ID"`
}

func VMMappingFromJSON(jsonString string) (*VMMapping, error) {
	var vmMapping *VMMapping
	err := json.Unmarshal([]byte(jsonString), &vmMapping)
	if err != nil {
		return nil, err
	}

	return vmMapping, nil
}

type PlacementResponse struct {
	VMS []VMMapping `json:"VMS"`
}

func (response *PlacementResponse) ToJSON() (string, error) {
	jsonBytes, err := json.Marshal(response)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (response *PlacementResponse) Equals(response2 *PlacementResponse) bool {
	if response2 == nil {
		return false
	}

	if len(response.VMS) != len(response2.VMS) {
		return false
	}

	for i := 0; i < len(response.VMS); i++ {
		if response.VMS[i].ID != response2.VMS[i].ID {
			return false
		}
		if response.VMS[i].HostID != response2.VMS[i].HostID {
			return false
		}
	}

	return true
}

func ParsePlacementResponse(jsonString string) (*PlacementResponse, error) {
	var response *PlacementResponse
	err := json.Unmarshal([]byte(jsonString), &response)
	if err != nil {
		return nil, err
	}

	return response, nil
}
