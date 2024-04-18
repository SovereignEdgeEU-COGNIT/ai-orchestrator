package opennebula

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const prometheusURL = "http://192.168.1.156:9090"

func TestGetHostIDs(t *testing.T) {
	hostIDs, err := GetHostIDs(prometheusURL)
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.NotEmpty(t, hostIDs, "hostIDs should not be empty")
}

func TestGetHostCPU(t *testing.T) {
	cpuLoad, err := GetHostCPU(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, cpuLoad > 0, "cpuLoad should be greater than 0")
}

func TestGetHostBusyCPU(t *testing.T) {
	cpuLoad, err := GetHostCPUBusy(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, cpuLoad > 0, "cpuLoad should be greater than 0")
}

func TestGetHostUsedMem(t *testing.T) {
	usedMem, err := GetHostUsedMem(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, usedMem > 0, "usedMem should be greater than 0")
}

func TestGetHostAvailMem(t *testing.T) {
	availMem, err := GetHostAvailMem(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, availMem > 0, "availMem should be greater than 0")
}

func TestGetHostNetTX(t *testing.T) {
	netTX, err := GetHostNetTX(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, netTX > 0, "netTrans should be greater than 0")
}

func TestGetHostNetRX(t *testing.T) {
	netRX, err := GetHostNetRX(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, netRX > 0, "netRX should be greater than 0")
}

func TestGetHostDiskRead(t *testing.T) {
	diskRead, err := GetHostDiskRead(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, diskRead >= 0, "diskRead should be greater than 0")
}

func TestGetHostDiskWrite(t *testing.T) {
	diskWrite, err := GetHostDiskWrite(prometheusURL, "4")
	assert.Nil(t, err)
	if err != nil {
		t.Error(err)
	}
	assert.True(t, diskWrite >= 0, "diskWrite should be greater than 0")
}
