package emulator

import (
	"fmt"
	"os"
	"strconv"
	"testing"
)

//const prometheusURL = "http://192.168.1.128:9090"

func getPrometheusURL() string {
	// add the env variable PROMETHEUS_HOST
	prometheusHost := os.Getenv("PROMETHEUS_HOST")

	prometheusPortStr := os.Getenv("PROMETHEUS_PORT")
	prometheusPort, _ := strconv.Atoi(prometheusPortStr)
	prometheusURL := "http://" + prometheusHost + ":" + strconv.Itoa(prometheusPort)
	return prometheusURL
}

func TestGetFlavourMetric(t *testing.T) {
	prometheusURL = getPrometheusURL()
	flavourMetric, err := GetFlavourMetricForHost(prometheusURL, "Cognit-test_emulated_host_1")
	if err != nil {
		t.Error(err)
	}
	fmt.Println("Name:", flavourMetric.Name)
	fmt.Println("Timestamp:", flavourMetric.Timestamp)
	fmt.Println("CPURate:", flavourMetric.CPURate)
	fmt.Println("MemoryUsage:", flavourMetric.MemoryUsage)
}
