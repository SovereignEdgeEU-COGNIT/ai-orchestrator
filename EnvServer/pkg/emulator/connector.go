package emulator

import (
	"encoding/json"
	"errors"
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/client"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/go-resty/resty/v2"
	log "github.com/sirupsen/logrus"
)

type SRInfo struct {
	IP       string   `json:"ip"`
	Name     string   `json:"name"`
	Port     int      `json:"port"`
	Flavor   string   `json:"flavor"`
	HostInfo HostInfo `json:"host_info"`
}

type HostInfo struct {
	IP   string `json:"ip"`
	Name string `json:"name"`
	Port int    `json:"port"`
}

type NodeType struct {
	Host HostInfo `json:"Host"`
	SR   SRInfo   `json:"SR"`
}

/* type EmulatedHost struct {
	HostInfo SRInfo `json:"host"`
	Flavors  []string `json:"flavors"`
} */

type EmulatorConnector struct {
	ctrlPlaneHost  string
	ctrlPlanePort  int
	envServerHost  string
	envServerPort  int
	prometheusHost string
	prometheusPort int
	client         *resty.Client
	envClient      *client.EnvClient
}

func CreateEmulatorConnector(ctrlPlaneHost string, ctrlPlanePort int, envServerHost string, envServerPort int, prometheusHost string, prometheusPort int) *EmulatorConnector {
	log.WithFields(log.Fields{
		"CtrlPlaneHost":  ctrlPlaneHost,
		"CtrlPlanePort":  ctrlPlanePort,
		"EnvServerHost":  envServerHost,
		"EnvServerPort":  envServerPort,
		"PrometheusHost": prometheusHost,
		"PrometheusPort": prometheusPort}).
		Info("Creating emulator connector")

	return &EmulatorConnector{
		ctrlPlaneHost:  ctrlPlaneHost,
		ctrlPlanePort:  ctrlPlanePort,
		envServerHost:  envServerHost,
		envServerPort:  envServerPort,
		prometheusHost: prometheusHost,
		prometheusPort: prometheusPort,
		client:         resty.New(),
		envClient:      client.CreateEnvClient(envServerHost, envServerPort, true)}
}

func checkStatus(statusCode int, body string) error {
	if statusCode != 200 {
		return errors.New(body)
	}

	return nil
}

func (c *EmulatorConnector) sync() error {
	prometheusURL := "http://" + c.prometheusHost + ":" + strconv.Itoa(c.prometheusPort)

	emulatedSRs, emulatedHosts, err := getEmulatorState(c)

	if emulatedHosts != nil {
		log.WithFields(log.Fields{"hosts": emulatedHosts}).Info("Emulated hosts currently unused, only partially implemented the SR and Host sync")
	}

	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error fetching emulator state")
		return err
	}

	hosts, err := c.envClient.GetHosts()
	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error fetching hosts from env server")
		return err
	}

	srMap := make(map[string]bool)
	for _, host := range hosts {
		srMap[host.HostID] = true
	}

	vms, err := c.envClient.GetVMs()
	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error fetching VMs from env server")
		return err
	}

	vmMaps := make(map[string]bool)
	for _, emulatedSR := range emulatedSRs {
		vmMaps[emulatedSR.Flavor] = true
		/* for _, flavor := range emulatedHost.Flavors {
		} */
	}

	// Add missing hosts
	for _, emulatedSR := range emulatedSRs {
		if _, ok := srMap[emulatedSR.Name]; !ok {
			log.WithFields(log.Fields{"host": emulatedSR.Name}).Info("Adding host to env server")
			host, err := c.envClient.GetHost(emulatedSR.Name)
			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error fetching host from env server")
				return err
			}
			if host == nil {
				totalCPU, err := GetTotalCPU(prometheusURL, emulatedSR.Name)
				if err != nil {
					log.WithFields(log.Fields{"error": err}).Error("Error fetching total CPU")
					return err
				}

				totalMemory, err := GetTotalMemory(prometheusURL, emulatedSR.Name)
				if err != nil {
					log.WithFields(log.Fields{"error": err}).Error("Error fetching total memory")
					return err
				}

				host := &core.Host{HostID: emulatedSR.Name, TotalCPU: totalCPU, TotalMemory: totalMemory}

				err = c.envClient.AddHost(host)
				if err != nil {
					log.WithFields(log.Fields{"error": err}).Error("Error adding host to env server")
					return err
				}
			}
		} else {
			log.WithFields(log.Fields{"host": emulatedSR.Name}).Info("updating host in env server")
		}
	}

	// Remove hosts that are not in the emulator
	for _, host := range hosts {
		if _, ok := srMap[host.HostID]; !ok {
			log.WithFields(log.Fields{"host": host.HostID}).Info("Removing host from env server")
			err = c.envClient.RemoveHost(host.HostID)
			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error removing host from env server")
				return err
			}

		}
	}

	// Add VMs
	for _, emulatedSR := range emulatedSRs {
		//for _, flavor := range emulatedSR.Flavors {
		name := emulatedSR.Name + "_" + emulatedSR.Flavor
		vm, err := c.envClient.GetVM(name)
		if err != nil {
			log.WithFields(log.Fields{"error": err}).Error("Error fetching VM from env server")
			return err
		}
		if vm == nil {
			vm := &core.VM{VMID: name}
			err = c.envClient.AddVM(vm)
			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error adding VM to env server")
				return err
			}

			err = c.envClient.Bind(vm.VMID, emulatedSR.Name)
			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error binding VM to host")
				return err
			}

			log.WithFields(log.Fields{"VMID": vm.VMID, "HostID": vm.HostID}).Info("Adding VM to env server")
		}
		//}
	}

	// Remove VMs that are not found in the emulator
	for _, vm := range vms {
		if _, ok := vmMaps[vm.VMID]; !ok {
			log.WithFields(log.Fields{"vm": vm.VMID}).Info("Removing unbinding VM")
			err = c.envClient.Unbind(vm.VMID, vm.HostID)
			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error unbinding VM")
				return err
			}
		}
	}

	return nil
}

func getEmulatorState(c *EmulatorConnector) ([]SRInfo, []HostInfo, error) {
	nodeTypes, err := listEmulatorNodes(c, "sr")

	if err != nil {
		return nil, nil, err
	}

	emulatedSRs := make([]SRInfo, len(nodeTypes))
	for i, nodeType := range nodeTypes {
		emulatedSRs[i] = nodeType.SR
	}

	nodeTypes, err = listEmulatorNodes(c, "host")

	if err != nil {
		return nil, nil, err
	}

	emulatedHosts := make([]HostInfo, len(nodeTypes))
	for i, nodeType := range nodeTypes {
		emulatedHosts[i] = nodeType.Host
	}

	return emulatedSRs, emulatedHosts, nil
}

func listEmulatorNodes(c *EmulatorConnector, nodeType string) ([]NodeType, error) {
	resp, err := c.client.R().
		Get("http://" + c.ctrlPlaneHost + ":" + strconv.Itoa(c.ctrlPlanePort) + "/list?node_type=" + nodeType)
	if err != nil {
		return nil, err
	}

	err = checkStatus(resp.StatusCode(), string(resp.Body()))
	if err != nil {
		return nil, err
	}

	var nodeTypes []NodeType
	err = json.Unmarshal(resp.Body(), &nodeTypes)
	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error unmarshalling response")
		return nil, err
	}
	return nodeTypes, nil
}

func (c *EmulatorConnector) fetchMetrics() error {
	prometheusURL := "http://" + c.prometheusHost + ":" + strconv.Itoa(c.prometheusPort)

	vms, err := c.envClient.GetVMs()
	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error fetching VMs from env server")
		return err
	}

	hosts, err := c.envClient.GetHosts()
	if err != nil {
		log.WithFields(log.Fields{"error": err}).Error("Error fetching hosts from env server")
		return err
	}

	for _, host := range hosts {
		hostMetric, err := GetFlavourMetricForHost(prometheusURL, host.HostID)

		if err == nil {
			err = c.envClient.AddMetric(host.HostID, core.HostType, &core.Metric{Timestamp: hostMetric.Timestamp, CPU: hostMetric.CPURate, Memory: hostMetric.MemoryUsage, DiskRead: hostMetric.DiskRead, DiskWrite: hostMetric.DiskWrite, NetRX: hostMetric.NetRx, NetTX: hostMetric.NetTx})

			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error adding metric to env server")
			}
		} else {
			log.WithFields(log.Fields{"error": err}).Error("Error fetching flavour metric for host")
		}
	}

	for _, vm := range vms {
		hostMetric, err := GetFlavourMetricForHost(prometheusURL, vm.HostID) // Assume one VM per host

		if err == nil {
			err = c.envClient.AddMetric(vm.VMID, core.VMType, &core.Metric{Timestamp: hostMetric.Timestamp, CPU: hostMetric.CPURate, Memory: hostMetric.MemoryUsage})

			if err != nil {
				log.WithFields(log.Fields{"error": err}).Error("Error adding metric to env server")
			}
		} else {
			log.WithFields(log.Fields{"error": err}).Error("Error fetching flavour metric for VM")
		}
	}

	if err != nil {
		return err
	} else {
		return nil
	}

}

func (c *EmulatorConnector) Start() {
	for {
		log.WithFields(log.Fields{"CtrlPlaneHost": c.ctrlPlaneHost, "CtrlPlanePort": c.ctrlPlanePort, "EnvServerHost": c.envServerHost, "EnvServerPort": c.envServerPort, "PrometheusHost": c.prometheusHost, "PrometheusPort": c.prometheusPort}).Info("Syncing")
		err := c.sync()
		if err != nil {
			log.WithFields(log.Fields{"error": err}).Error("Error syncing data")
		}

		err = c.fetchMetrics()
		if err != nil {
			log.WithFields(log.Fields{"error": err}).Error("Error fetching metrics")
		}

		time.Sleep(1 * time.Second)
	}
}
