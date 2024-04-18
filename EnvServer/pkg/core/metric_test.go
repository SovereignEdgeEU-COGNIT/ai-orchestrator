package core

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestMetricToJSON(t *testing.T) {
	metric := &Metric{Memory: 1, CPU: 2, Timestamp: time.Now()}

	jsonStr, err := metric.ToJSON()
	assert.Nil(t, err)

	_, err = ConvertJSONToMetric(jsonStr + "error")
	assert.NotNil(t, err)

	metric2, err := ConvertJSONToMetric(jsonStr)
	assert.Nil(t, err)
	assert.True(t, metric2.Equals(metric))
}

func TestMetricArrayToJSON(t *testing.T) {
	metrics := []*Metric{}

	metric1 := &Metric{Memory: 1, CPU: 2, Timestamp: time.Now()}
	metric2 := &Metric{Memory: 3, CPU: 4, Timestamp: time.Now()}

	metrics = append(metrics, metric1)
	metrics = append(metrics, metric2)

	jsonStr, err := ConvertMetricArrayToJSON(metrics)
	assert.Nil(t, err)

	_, err = ConvertJSONToMetricArray(jsonStr + "error")
	assert.NotNil(t, err)

	metrics2, err := ConvertJSONToMetricArray(jsonStr)
	assert.Nil(t, err)
	assert.True(t, IsMetricArraysEqual(metrics, metrics2))

}
