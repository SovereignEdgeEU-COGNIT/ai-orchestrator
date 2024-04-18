package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHostToJSON(t *testing.T) {
	host := &Host{HostID: "1"}

	jsonStr, err := host.ToJSON()
	assert.Nil(t, err)

	_, err = ConvertJSONToHost(jsonStr + "error")
	assert.NotNil(t, err)

	host2, err := ConvertJSONToHost(jsonStr)
	assert.Nil(t, err)
	assert.True(t, host2.Equals(host))
}

func TestHostArrayToJSON(t *testing.T) {
	var hosts []*Host

	host1 := &Host{HostID: "1", StateID: 1, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	host2 := &Host{HostID: "2", StateID: 2, TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}

	hosts = append(hosts, host1)
	hosts = append(hosts, host2)

	jsonStr, err := ConvertHostArrayToJSON(hosts)
	assert.Nil(t, err)

	_, err = ConvertJSONToHostArray(jsonStr + "error")
	assert.NotNil(t, err)

	hosts2, err := ConvertJSONToHostArray(jsonStr)
	assert.Nil(t, err)
	assert.True(t, IsHostArraysEqual(hosts, hosts2))
}
