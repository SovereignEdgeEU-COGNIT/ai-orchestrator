package database

import (
	"testing"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestHostMetrics(t *testing.T) {
	db, err := PrepareTests()
	assert.Nil(t, err)
	defer db.Close()

	err = db.AddMetric("host1", core.HostType, &core.Metric{Timestamp: time.Unix(int64(1672531200), 0), CPU: 1, Memory: 10})
	assert.Nil(t, err)
	err = db.AddMetric("host2", core.HostType, &core.Metric{Timestamp: time.Unix(int64(1672531201), 0), CPU: 100, Memory: 4000})
	assert.Nil(t, err)
	err = db.AddMetric("vm1", core.VMType, &core.Metric{Timestamp: time.Unix(int64(1672531201), 0), CPU: 1000, Memory: 40000})
	assert.Nil(t, err)
	err = db.AddMetric("host1", core.HostType, &core.Metric{Timestamp: time.Unix(int64(1672531201), 0), CPU: 2, Memory: 20})
	assert.Nil(t, err)
	err = db.AddMetric("host1", core.HostType, &core.Metric{Timestamp: time.Unix(int64(1672531202), 0), CPU: 3, Memory: 30})
	assert.Nil(t, err)
	err = db.AddMetric("host1", core.HostType, &core.Metric{Timestamp: time.Unix(int64(1672531203), 0), CPU: 4, Memory: 40})
	assert.Nil(t, err)

	metrics, err := db.GetMetrics("host1", core.HostType, time.Unix(int64(1672531100), 0), 1)
	assert.Nil(t, err)
	assert.Len(t, metrics, 1)
	assert.Equal(t, int64(1672531200), metrics[0].Timestamp.Unix())
	assert.Equal(t, float64(1), metrics[0].CPU)
	assert.Equal(t, float64(10), metrics[0].Memory)

	metrics, err = db.GetMetrics("host1", core.HostType, time.Unix(int64(1672531200), 0), 1)
	assert.Nil(t, err)
	assert.Len(t, metrics, 1)
	assert.Equal(t, int64(1672531201), metrics[0].Timestamp.Unix())
	assert.Equal(t, float64(2), metrics[0].CPU)
	assert.Equal(t, float64(20), metrics[0].Memory)

	metrics, err = db.GetMetrics("vm1", core.VMType, time.Unix(int64(1672531200), 0), 1)
	assert.Nil(t, err)
	assert.Len(t, metrics, 1)
	assert.Equal(t, int64(1672531201), metrics[0].Timestamp.Unix())
	assert.Equal(t, float64(1000), metrics[0].CPU)
	assert.Equal(t, float64(40000), metrics[0].Memory)

	metrics, err = db.GetMetrics("host1", core.HostType, time.Unix(int64(1672531100), 0), 400)
	assert.Nil(t, err)
	assert.Len(t, metrics, 4)

	metrics, err = db.GetMetrics("host2", core.HostType, time.Unix(int64(1672531100), 0), 400)
	assert.Nil(t, err)
	assert.Len(t, metrics, 1)
}
