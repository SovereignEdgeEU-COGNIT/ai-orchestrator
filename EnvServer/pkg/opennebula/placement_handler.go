package opennebula

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"

	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

func (server *IntegrationServer) handlePlacementRequest(c *gin.Context) {
	jsonData, err := ioutil.ReadAll(c.Request.Body)
	if err != nil {
		fmt.Println(err)
		log.Error("Error reading placement request: ", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	log.Info("Placement request received: ", string(jsonData))

	placementRequest, err := ParsePlacementRequest(string(jsonData))
	if err != nil {
		fmt.Println(err)
		log.Error("Error parsing placement request: ", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	vms := placementRequest.VMs

	vmMapping, err := server.offloadVMPlacement(&vms)
	if err != nil {
		fmt.Println(err)
		log.Error("Error placing VMs: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// vmMapping := make([]VMMapping, len(vms))

	// for i, vm := range vms {
	// 	hosts := vm.HostIDs
	// 	if len(hosts) == 0 {
	// 		log.Error("Invalid placement request, no hosts available")
	// 		continue
	// 	}

	// 	randomIndex := rand.Intn(len(hosts))
	// 	randomHost := hosts[randomIndex]

	// 	vmMapping[i] = VMMapping{ID: vm.ID, HostID: randomHost}
	// }

	placementResponse := PlacementResponse{VMS: vmMapping}
	respJSON, err := placementResponse.ToJSON()
	if err != nil {
		fmt.Println(err)
		log.Error("Error marshalling placement response: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Info("Placement response: ", string(respJSON))

	c.Data(http.StatusOK, "application/json", []byte(respJSON))
}

func (server *IntegrationServer) placeVMs(vms []VM) ([]VMMapping, error) {

	vmMappings := make([]VMMapping, len(vms))

	for i, vm := range vms {
		vmMapping, err := server.randomVMPlacement(vm)

		if err != nil {
			log.Error("Error placing VM: ", err)
			continue
		}

		vmMappings[i] = vmMapping
	}

	return vmMappings, nil
}

func (server *IntegrationServer) randomVMPlacement(vm VM) (VMMapping, error) {
	hosts := vm.HostIDs

	if len(hosts) == 0 {
		return VMMapping{ID: vm.ID, HostID: -1}, fmt.Errorf("Invalid placement request, no hosts available")
	}

	randomIndex := rand.Intn(len(hosts))
	randomHost := hosts[randomIndex]

	return VMMapping{ID: vm.ID, HostID: randomHost}, nil
}

func (server *IntegrationServer) offloadVMPlacement(vms *[]VM) ([]VMMapping, error) {
	vmMappings := make([]VMMapping, len(*vms))

	for i, vm := range *vms {
		vmMapping, err := server.MLClient.PlaceVM(&vm)

		if err != nil {
			fmt.Println(err)
			log.Error("Error placing VM: ", err)
			continue
		}

		vmMappings[i] = *vmMapping
	}

	return vmMappings, nil
}
