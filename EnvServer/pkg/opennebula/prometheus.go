package opennebula

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type PrometheusResponse struct {
	Status string `json:"status"`
	Data   Data   `json:"data"`
}

type Data struct {
	ResultType string   `json:"resultType"`
	Result     []Result `json:"result"`
}

type Result struct {
	Metric Metric        `json:"metric"`
	Value  []interface{} `json:"value"`
}

type Metric struct {
	Device    string `json:"device"`
	Name      string `json:"__name__"`
	Instance  string `json:"instance"`
	Job       string `json:"job"`
	OneHostID string `json:"one_host_id"`
	OneVMID   string `json:"one_vm_id"`
}

func QueryPrometheus(prometheusURL, query string) ([]byte, error) {
	fullURL := fmt.Sprintf("%s/api/v1/query?query=%s", prometheusURL, url.QueryEscape(query))

	resp, err := http.Get(fullURL)
	if err != nil {
		return []byte{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return []byte{}, err
	}

	return body, nil
}

func GetHostIDs(prometheusURL string) ([]string, error) {
	var hostIDs []string

	query := `opennebula_host_state`
	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return hostIDs, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return hostIDs, err
	}

	for _, result := range resp.Data.Result {
		hostIDs = append(hostIDs, result.Metric.OneHostID)
	}

	return hostIDs, nil
}

//increase(opennebula_libvirt_cpu_seconds_total[30s])
/*
func GetVMIDs(prometheusURL string) ([]string, error) {
	var vmsIDs []string

	query := `opennebula_vm_state`
	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return vmsIDs, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return vmsIDs, err
	}

	for _, result := range resp.Data.Result {
		vmsIDs = append(vmsIDs, result.Metric.OneVMID)
	}

	return vmsIDs, nil
}
*/
func GetHostTotalMem(prometheusURL, hostID string) ([]string, error) {
	var vmsIDs []string

	query := `opennebula_host_vms{one_host_id=` + hostID + `}`
	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return vmsIDs, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return vmsIDs, err
	}

	return vmsIDs, nil
}

func GetHostCPU(prometheusURL, hostID string) (float64, error) {
	query := `sum by (one_host_id)(rate(node_cpu_seconds_total{mode='user',one_host_id="` + hostID + `"}[40s])) * 100`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}
	return strconv.ParseFloat(str, 64)
}

func GetHostCPUBusy(prometheusURL, hostID string) (float64, error) {
	query := `(((count(count(node_cpu_seconds_total{one_host_id="` + hostID + `"}) by (cpu))) - avg(sum by (mode)(rate(node_cpu_seconds_total{mode='idle',one_host_id="` + hostID + `"}[300s])))) * 100) / count(count(node_cpu_seconds_total{one_host_id="` + hostID + `"}) by (cpu))`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	return strconv.ParseFloat(str, 64)
}

func GetHostUsedMem(prometheusURL, hostID string) (float64, error) {
	query := `node_memory_MemFree_bytes{one_host_id="` + hostID + `"}`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	memBytes, err := strconv.ParseFloat(str, 64)
	if err != nil {
		return 0.0, err
	}
	memMBInt64 := (memBytes / 1024 / 1024)

	return memMBInt64, nil
}

func GetHostAvailMem(prometheusURL, hostID string) (int64, error) {
	query := `node_memory_MemTotal_bytes{one_host_id="` + hostID + `"}`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	memBytes, err := strconv.ParseFloat(str, 64)
	if err != nil {
		return 0.0, err
	}
	memMBInt64 := int64(memBytes / 1024 / 1024)

	return memMBInt64, nil
}

func GetHostNetTX(prometheusURL, hostID string) (float64, error) {
	query := `rate(node_network_transmit_bytes_total{one_host_id="` + hostID + `"}[40s])*8`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	var sumIfs float64
	for _, result := range resp.Data.Result {
		if strings.HasPrefix(result.Metric.Device, "br") {
			continue
		}

		sumIfsStr := result.Value[1].(string)
		ifs, err := strconv.ParseFloat(sumIfsStr, 64)
		if err != nil {
			return 0.0, err
		}
		sumIfs += ifs
	}

	return sumIfs, nil
}

func GetHostNetRX(prometheusURL, hostID string) (float64, error) {
	query := `rate(node_network_receive_bytes_total{one_host_id="` + hostID + `"}[40s])*8`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	var sumIfs float64
	for _, result := range resp.Data.Result {
		if strings.HasPrefix(result.Metric.Device, "br") {
			continue
		}

		sumIfsStr := result.Value[1].(string)
		ifs, err := strconv.ParseFloat(sumIfsStr, 64)
		if err != nil {
			return 0.0, err
		}
		sumIfs += ifs
	}

	return sumIfs, nil
}

func GetHostDiskRead(prometheusURL, hostID string) (float64, error) {
	query := `sum(rate(node_disk_read_bytes_total{one_host_id="` + hostID + `"}[40s]))`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	diskRead, err := strconv.ParseFloat(str, 64)

	return diskRead, nil
}

func GetHostDiskWrite(prometheusURL, hostID string) (float64, error) {
	query := `sum(rate(node_disk_written_bytes_total{one_host_id="` + hostID + `"}[40s]))`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	diskRead, err := strconv.ParseFloat(str, 64)

	return diskRead, nil
}

func GetHostEnergyUsage(prometheusURL, hostID string) (float64, error) {
	query := `sum(rate(scaph_host_power_microwatts{host_id="` + hostID + `"}[40s]))`

	r, err := QueryPrometheus(prometheusURL, query)
	if err != nil {
		return 0.0, err
	}

	var resp PrometheusResponse
	err = json.Unmarshal(r, &resp)
	if err != nil {
		return 0.0, err
	}

	if len(resp.Data.Result) == 0 {
		return 0.0, fmt.Errorf("No data found, hostID: %s", hostID)
	}

	if len(resp.Data.Result[0].Value) < 2 {
		return 0.0, fmt.Errorf("No value found, hostID: %s", hostID)
	}

	str, ok := resp.Data.Result[0].Value[1].(string)
	if !ok {
		return 0.0, fmt.Errorf("Failed to convert value to string")
	}

	energyUsage, err := strconv.ParseFloat(str, 64)

	return energyUsage, nil
}
