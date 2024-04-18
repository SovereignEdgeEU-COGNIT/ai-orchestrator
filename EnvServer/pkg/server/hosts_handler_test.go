package server

import (
	"testing"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestAddHost(t *testing.T) {
	client, server, done := prepareTests(t)

	host, err := client.GetHost("host1")
	assert.Nil(t, err)
	assert.Nil(t, host)

	host = &core.Host{HostID: "host1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddHost(host)
	assert.Nil(t, err)

	host, err = client.GetHost("host1")
	assert.Nil(t, err)
	assert.NotNil(t, host)
	assert.Equal(t, "host1", host.HostID)

	server.Shutdown()
	<-done
}

func TestGetHost(t *testing.T) {
	client, server, done := prepareTests(t)

	host, err := client.GetHost("host1")
	assert.Nil(t, err)
	assert.Nil(t, host)

	host = &core.Host{HostID: "host1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddHost(host)
	assert.Nil(t, err)

	host, err = client.GetHost("host1")
	assert.Nil(t, err)
	assert.NotNil(t, host)
	assert.Equal(t, "host1", host.HostID)

	server.Shutdown()
	<-done
}

func TestGetHosts(t *testing.T) {
	client, server, done := prepareTests(t)

	hosts, err := client.GetHosts()
	assert.Nil(t, err)
	assert.Nil(t, hosts)

	host := &core.Host{HostID: "host1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddHost(host)
	assert.Nil(t, err)

	host = &core.Host{HostID: "host2", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err = client.AddHost(host)
	assert.Nil(t, err)

	hosts, err = client.GetHosts()
	assert.Nil(t, err)
	assert.NotNil(t, hosts)
	assert.Len(t, hosts, 2)

	server.Shutdown()
	<-done
}

func TestRemoveHost(t *testing.T) {
	client, server, done := prepareTests(t)

	host := &core.Host{HostID: "host1", TotalCPU: 1600, TotalMemory: 16785711104, UsageCPU: 800, UsageMemory: 8385855552}
	err := client.AddHost(host)
	assert.Nil(t, err)

	host, err = client.GetHost("host1")
	assert.Nil(t, err)
	assert.NotNil(t, host)
	assert.Equal(t, "host1", host.HostID)

	err = client.RemoveHost("host1")
	assert.Nil(t, err)

	host, err = client.GetHost("host1")
	assert.Nil(t, err)
	assert.Nil(t, host)

	server.Shutdown()
	<-done
}
