package core

import (
	"encoding/json"
	"time"
)

type Metric struct {
	Timestamp   time.Time `json:"timestamp"`
	CPU         float64   `json:"cpu"`
	Memory      float64   `json:"memory"`
	DiskRead    float64   `json:"disk_read"`
	DiskWrite   float64   `json:"disk_write"`
	NetRX       float64   `json:"netrx"`
	NetTX       float64   `json:"nettx"`
	EnergyUsage float64   `json:"energy_usage"`
}

func ConvertJSONToMetric(jsonString string) (*Metric, error) {
	var metric *Metric
	err := json.Unmarshal([]byte(jsonString), &metric)
	if err != nil {
		return nil, err
	}

	return metric, nil
}

func ConvertJSONToMetricArray(jsonString string) ([]*Metric, error) {
	var metrics []*Metric

	err := json.Unmarshal([]byte(jsonString), &metrics)
	if err != nil {
		return metrics, err
	}

	return metrics, nil
}

func ConvertMetricArrayToJSON(metrics []*Metric) (string, error) {
	jsonBytes, err := json.Marshal(metrics)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (metric *Metric) ToJSON() (string, error) {
	jsonBytes, err := json.Marshal(metric)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

func (metric *Metric) Equals(metric2 *Metric) bool {
	if metric2 == nil {
		return false
	}

	if metric.Memory == metric2.Memory &&
		metric.CPU == metric2.CPU &&
		metric.DiskRead == metric2.DiskRead &&
		metric.DiskWrite == metric2.DiskWrite &&
		metric.NetRX == metric2.NetRX &&
		metric.NetTX == metric2.NetTX &&
		metric.Timestamp.UnixNano() == metric2.Timestamp.UnixNano() {
		return true
	}

	return false
}

func IsMetricArraysEqual(metrics []*Metric, metrics2 []*Metric) bool {
	if len(metrics) != len(metrics2) {
		return false
	}

	for i := 0; i < len(metrics); i++ {
		if !metrics[i].Equals(metrics2[i]) {
			return false
		}
	}

	return true
}
