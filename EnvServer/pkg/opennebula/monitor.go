package opennebula

import (
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/client"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	log "github.com/sirupsen/logrus"
)

type monitor struct {
	prometheusURL string
	stopFlag      bool
	client        *client.EnvClient
}

func newMonitor(prometheusURL string) *monitor {
	m := &monitor{}
	m.prometheusURL = prometheusURL
	m.client = client.CreateEnvClient("localhost", 50080, true)
	return m
}

func (m *monitor) runForever() {
	go func() {
		for {
			if m.stopFlag {
				return
			}

			hostIDs, err := GetHostIDs(m.prometheusURL)
			if err != nil {
				log.Error("Failed to get host IDs: ", err)
			}

			time.Sleep(1 * time.Second)

			oneHosts := make(map[string]bool)
			var cpuTotal float64
			for _, hostID := range hostIDs {
				// Add hosts to the envserver that are present on opennebula
				hostFromServer, err := m.client.GetHost(hostID)
				if err != nil {
					log.Debug("Failed to get host from server: ", err)
				}

				oneHosts[hostID] = true
				if hostFromServer == nil {
					host := &core.Host{HostID: hostID}
					err = m.client.AddHost(host)
					if err != nil {
						log.Debug("Failed to add host to server: ", err)
					}
				}

				// Collect metric data
				now := time.Now()

				cpuTotal, err = GetHostCPU(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host CPU: ", err)
				}
				usedMem, err := GetHostUsedMem(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host used memory: ", err)
				}
				diskRead, err := GetHostDiskRead(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host available memory: ", err)
				}
				diskWrite, err := GetHostDiskWrite(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host available memory: ", err)
				}
				netRX, err := GetHostNetRX(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host available memory: ", err)
				}
				netTX, err := GetHostNetRX(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host available memory: ", err)
				}
				energyUsage, err := GetHostEnergyUsage(m.prometheusURL, hostID)
				if err != nil {
					log.Debug("Failed to get host energy usage: ", err)
				}

				// Add metric to the envserver
				err = m.client.AddMetric(hostID, core.HostType, &core.Metric{Timestamp: now, CPU: cpuTotal, Memory: usedMem, DiskRead: diskRead, DiskWrite: diskWrite, NetTX: netTX, NetRX: netRX, EnergyUsage: energyUsage})
				if err != nil {
					log.Error("Failed to add metric to server: ", err)
				}
			}

			serverHosts, err := m.client.GetHosts()
			if err != nil {
				log.Error("Failed to get hosts from server: ", err)
			}

			// TODO: This code is untested
			for _, serverHost := range serverHosts {
				_, ok := oneHosts[serverHost.HostID]
				if !ok {
					err := m.client.RemoveHost(serverHost.HostID)
					if err != nil {
						log.Error("Failed to remove host from server: ", err)
					}
				}
			}

			err = m.updateVMMetrics()
			if err != nil {
				log.Error("Failed to update VM metrics: ", err)
			}
		}
	}()
}

func (m *monitor) updateVMMetrics() error {

	vmIDs, err := GetVMIDs(m.prometheusURL)

	if err != nil {
		return err
	}

	vmMap := make(map[string]*core.VM)
	for _, vmID := range vmIDs {
		vm := &core.VM{
			VMID: vmID,
		}
		vmMap[vmID] = vm
	}

	if err = MapVMHostIDs(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsDiskRead(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsDiskWrite(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsMemUsage(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsMemTotal(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsCPUTotal(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsCPUUsage(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsNetRx(m.prometheusURL, vmMap); err != nil {
		return err
	}

	if err = GetVMsNetTx(m.prometheusURL, vmMap); err != nil {
		return err
	}

	vmsServer, err := m.client.GetVMs()

	if err != nil {
		return err
	}

	for id, vm := range vmMap {

		//if vm is not in the server, add it
		inServer := false

		for _, serverVM := range vmsServer {

			if serverVM.VMID == id {
				inServer = true

				if serverVM.HostID != vm.HostID {
					err = m.client.Bind(vm.VMID, vm.HostID)
					if err != nil {
						return err
					}
				}
			}
		}

		if !inServer {
			err = m.client.AddVM(vm)

			if err != nil {
				return err
			}
		}

		err = m.client.AddMetric(vm.VMID, core.VMType, &core.Metric{Timestamp: time.Now(), CPU: vm.UsageCPU, Memory: vm.UsageMemory, DiskRead: vm.DiskRead, DiskWrite: vm.DiskWrite, NetRX: vm.NetRX, NetTX: vm.NetTX})

		if err != nil {
			log.Error(err)
		}
	}

	return nil
}

func (m *monitor) stop() {
	m.stopFlag = true
}
