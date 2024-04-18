package database

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestAddHost(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	host, err := db.GetHost("test_host_id")
	assert.Nil(t, err)
	assert.Nil(t, host)

	host = &core.Host{HostID: "test_host_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	hosts, err := db.GetHosts()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(hosts))

	host, err = db.GetHost("test_host_id")
	assert.Nil(t, err)
	assert.NotNil(t, host)
	assert.Equal(t, "test_host_id", host.HostID)
}

func TestHostStateMetric(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	host := &core.Host{HostID: "test_host1_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	host = &core.Host{HostID: "test_host2_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	hosts, err := db.GetHosts()
	assert.Nil(t, err)

	counter := 0
	for _, host := range hosts {
		counter += host.StateID
	}
	assert.Equal(t, 3, counter)

	// If we remove a host and add another one, the state ID should be reused
	err = db.RemoveHost("test_host1_id")
	assert.Nil(t, err)

	host = &core.Host{HostID: "test_host3_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	hosts, err = db.GetHosts()
	assert.Nil(t, err)

	counter = 0
	for _, host := range hosts {
		counter += host.StateID
	}
	assert.Equal(t, 3, counter)
}

func TestSetHostResources(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	host := &core.Host{HostID: "test_host_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 1, UsageMemory: 2, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	err = db.SetVMResources(host.HostID, float64(1), float64(2), float64(3), float64(4), float64(5), float64(6))
	assert.Nil(t, err)

	hosts, err := db.GetHosts()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(hosts))
	assert.Equal(t, float64(1), hosts[0].UsageCPU)
	assert.Equal(t, float64(2), hosts[0].UsageMemory)
}

func TestRemoveHost(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	host := &core.Host{HostID: "test_host_id", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552, DiskRead: 1, DiskWrite: 2, NetRX: 3, NetTX: 4, EnergyUsage: 5}
	err = db.AddHost(host)
	assert.Nil(t, err)

	hosts, err := db.GetHosts()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(hosts))

	err = db.RemoveHost(host.HostID)
	assert.Nil(t, err)

	hosts, err = db.GetHosts()
	assert.Nil(t, err)
	assert.Equal(t, 0, len(hosts))
}
