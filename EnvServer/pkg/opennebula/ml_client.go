package opennebula

import (
	"errors"
	"strconv"

	"github.com/go-resty/resty/v2"
)

type MLClient struct {
	restyClient *resty.Client
	host        string
	port        int
	protocol    string
}

func CreateMLClient(host string, port int, insecure bool) *MLClient {
	client := &MLClient{}
	client.restyClient = resty.New()

	client.host = host
	client.port = port

	client.protocol = "https"
	if insecure {
		client.protocol = "http"
	}

	print("MLClient created with host: ", host, " port: ", port, " protocol: ", client.protocol)
	return client
}

func checkStatus(statusCode int, body string) error {
	if statusCode != 200 {
		return errors.New(body)
	}

	return nil
}

func (client *MLClient) PlaceVM(vm *VM) (*VMMapping, error) {
	jsonStr, err := vm.ToJSON()
	if err != nil {
		return nil, err
	}

	resp, err := client.restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(jsonStr).
		Post(client.protocol + "://" + client.host + ":" + strconv.Itoa(client.port) + "/api/place")
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	mapping, err := VMMappingFromJSON(string(resp.Body()))

	if err != nil {
		return nil, err
	}

	return mapping, nil
}
