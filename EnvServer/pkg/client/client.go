package client

import (
	"errors"
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/go-resty/resty/v2"
)

type EnvClient struct {
	restyClient *resty.Client
	host        string
	port        int
	protocol    string
}

func CreateEnvClient(host string, port int, insecure bool) *EnvClient {
	client := &EnvClient{}
	client.restyClient = resty.New()

	client.host = host
	client.port = port

	client.protocol = "https"
	if insecure {
		client.protocol = "http"
	}

	return client
}

func checkStatus(statusCode int, body string) error {
	if statusCode != 200 {
		return errors.New(body)
	}

	return nil
}

func (client *EnvClient) AddMetric(id string, metricType int, metric *core.Metric) error {
	jsonStr, err := metric.ToJSON()
	if err != nil {
		return err
	}

	resp, err := client.restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(jsonStr).
		Post(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/metrics?id=" + id + "&metrictype=" + strconv.Itoa(metricType))
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) GetMetrics(hostID string, metricType int, since time.Time, count int) ([]*core.Metric, error) {
	sinceUnixNano := since.UnixNano()

	resp, err := client.restyClient.R().
		Get(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/metrics?hostid=" + hostID + "&metrictype=" + strconv.Itoa(metricType) + "&since=" + strconv.FormatInt(sinceUnixNano, 10) + "&count=" + strconv.Itoa(count))
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	respBodyStr := string(resp.Body())

	metrics, err := core.ConvertJSONToMetricArray(respBodyStr)
	if err != nil {
		return nil, err
	}

	return metrics, nil
}

func (client *EnvClient) AddHost(host *core.Host) error {
	jsonStr, err := host.ToJSON()
	if err != nil {
		return err
	}

	resp, err := client.restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(jsonStr).
		Post(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/hosts")
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) GetHost(hostID string) (*core.Host, error) {
	resp, err := client.restyClient.R().
		Get(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/hosts/" + hostID)
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	respBodyStr := string(resp.Body())

	host, err := core.ConvertJSONToHost(respBodyStr)
	if err != nil {
		return nil, err
	}

	return host, nil
}

func (client *EnvClient) GetHosts() ([]*core.Host, error) {
	resp, err := client.restyClient.R().
		Get(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/hosts")
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	respBodyStr := string(resp.Body())

	hosts, err := core.ConvertJSONToHostArray(respBodyStr)
	if err != nil {
		return nil, err
	}

	return hosts, nil
}

func (client *EnvClient) RemoveHost(hostID string) error {
	resp, err := client.restyClient.R().
		Delete(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/hosts/" + hostID)
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) AddVM(vm *core.VM) error {
	jsonStr, err := vm.ToJSON()
	if err != nil {
		return err
	}

	resp, err := client.restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(jsonStr).
		Post(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms")
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) GetVM(vmID string) (*core.VM, error) {
	resp, err := client.restyClient.R().
		Get(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms/" + vmID)
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	respBodyStr := string(resp.Body())

	vm, err := core.ConvertJSONToVM(respBodyStr)
	if err != nil {
		return nil, err
	}

	return vm, nil
}

func (client *EnvClient) GetVMs() ([]*core.VM, error) {
	resp, err := client.restyClient.R().
		Get(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms")
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	respBodyStr := string(resp.Body())

	vms, err := core.ConvertJSONToVMArray(respBodyStr)
	if err != nil {
		return nil, err
	}

	return vms, nil
}

func (client *EnvClient) RemoveVM(vmID string) error {
	resp, err := client.restyClient.R().
		Delete(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms/" + vmID)
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) Bind(vmID, hostID string) error {
	resp, err := client.restyClient.R().
		Put(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms/" + vmID + "/" + hostID)
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}

func (client *EnvClient) Unbind(vmID, hostID string) error {
	resp, err := client.restyClient.R().
		Delete(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/vms/" + vmID + "/" + hostID)
	if err != nil {
		return err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return err
	}

	return nil
}
