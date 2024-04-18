package server

import (
	"io"
	"net/http"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

func (server *EnvServer) handleAddVMRequest(c *gin.Context) {
	jsonData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to read request body")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	vm, err := core.ConvertJSONToVM(string(jsonData))
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to convert JSON to VM")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	err = server.db.AddVM(vm)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to add VM to database")
		c.String(http.StatusInternalServerError, err.Error())
		return
	}

	log.WithFields(log.Fields{"VMID": vm.VMID}).Debug("Adding VM to database")
	c.String(http.StatusOK, "")
}

func (server *EnvServer) handleGetVMRequest(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		log.WithFields(log.Fields{"Error": "Paramater id must be specified"}).Error("Failed to get VM from database")
		c.String(http.StatusBadRequest, "Paramater id must be specified")
		return
	}

	if vm, err := server.db.GetVM(id); err != nil {
		log.WithFields(log.Fields{"Error": err, "VMID": id}).Error("Failed to get VM from database")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		log.WithFields(log.Fields{"VMID": id}).Debug("Getting VM from database")
		c.JSON(http.StatusOK, vm)
	}
}

func (server *EnvServer) handleGetVMsRequest(c *gin.Context) {
	if vms, err := server.db.GetVMs(); err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to get VMs from database")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		log.Debug("Getting VMs from database")
		c.JSON(http.StatusOK, vms)
	}
}

func (server *EnvServer) handleRemoveVMRequest(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		log.WithFields(log.Fields{"Error": "Paramater id must be specified"}).Error("Failed to remove VM from database")
		c.String(http.StatusBadRequest, "Paramater id must be specified")
		return
	}

	if err := server.db.RemoveVM(id); err != nil {
		log.WithFields(log.Fields{"Error": err, "VMID": id}).Error("Failed to remove VM from database")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		log.WithFields(log.Fields{"VMID": id}).Debug("Removing VM from database")
		c.String(http.StatusOK, "")
	}
}

func (server *EnvServer) handleBindRequest(c *gin.Context) {
	vmID := c.Param("id")
	if vmID == "" {
		log.WithFields(log.Fields{"Error": "Paramater id (vmId) must be specified"}).Error("Failed to bind VM to host")
		c.String(http.StatusBadRequest, "Paramater id (vmId) must be specified")
		return
	}
	hostID := c.Param("hostid")
	if hostID == "" {
		log.WithFields(log.Fields{"Error": "Paramater hostid must be specified"}).Error("Failed to bind VM to host")
		c.String(http.StatusBadRequest, "Paramater hostid must be specified")
		return
	}

	err := server.db.Bind(vmID, hostID)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to bind VM to host")
		c.String(http.StatusInternalServerError, err.Error())
		return
	}

	log.WithFields(log.Fields{"VMID": vmID, "HostID": hostID}).Debug("Binding VM to host")
}

func (server *EnvServer) handleUnbindRequest(c *gin.Context) {
	vmID := c.Param("id")
	if vmID == "" {
		log.WithFields(log.Fields{"Error": "Paramater id (vmId) must be specified"}).Error("Failed to unbind VM from host")
		c.String(http.StatusBadRequest, "Paramater id (vmId) must be specified")
		return
	}
	hostID := c.Param("hostid")
	if hostID == "" {
		log.WithFields(log.Fields{"Error": "Paramater hostid must be specified"}).Error("Failed to unbind VM from host")
		c.String(http.StatusBadRequest, "Paramater hostid must be specified")
		return
	}

	err := server.db.Unbind(vmID)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to unbind VM from host")
		c.String(http.StatusInternalServerError, err.Error())
		return
	}

	log.WithFields(log.Fields{"VMID": vmID, "HostID": hostID}).Debug("Unbinding VM from host")
}
