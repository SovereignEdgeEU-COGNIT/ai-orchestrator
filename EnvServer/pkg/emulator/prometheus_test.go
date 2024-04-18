package emulator

import (
	"fmt"
	"testing"
)

const prometheusURL = "http://localhost:9090"

func TestGetFlavourMetric(t *testing.T) {
	flavourMetric, err := GetFlavourMetricForHost(prometheusURL, "Cognit-test_emulated_host_1")
	if err != nil {
		t.Error(err)
	}
	fmt.Println("Name:", flavourMetric.Name)
	fmt.Println("Timestamp:", flavourMetric.Timestamp)
	fmt.Println("CPURate:", flavourMetric.CPURate)
	fmt.Println("MemoryUsage:", flavourMetric.MemoryUsage)
}
