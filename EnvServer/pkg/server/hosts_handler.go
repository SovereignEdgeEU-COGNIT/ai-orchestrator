package server

import (
	"io"
	"net/http"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

func (server *EnvServer) handleAddHostRequest(c *gin.Context) {
	jsonData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to read request body")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	host, err := core.ConvertJSONToHost(string(jsonData))
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to convert JSON to host")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	err = server.db.AddHost(host)
	if err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to add host")
		c.String(http.StatusInternalServerError, err.Error())
		return
	}

	log.WithFields(log.Fields{"Host": host}).Debug("Adding host")

	c.String(http.StatusOK, "")
}

func (server *EnvServer) calcNumberOfVMsPerHost(hostID string) (int, error) {
	var numberOfVMs int
	vms, err := server.db.GetVMs()
	if err != nil {
		return 0, err
	}

	for _, vm := range vms {
		if vm.HostID == hostID {
			numberOfVMs++
		}
	}
	return numberOfVMs, nil
}

func (server *EnvServer) handleGetHostRequest(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		log.WithFields(log.Fields{"Error": "Paramater id must be specified"}).Error("Failed to get host")
		c.String(http.StatusBadRequest, "Paramater id must be specified")
		return
	}

	if host, err := server.db.GetHost(id); err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to get host")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		if host != nil {
			host.VMs, err = server.calcNumberOfVMsPerHost(host.HostID)
			log.WithFields(log.Fields{"HostId": host.HostID}).Debug("Getting host from database")
		} else {
			log.WithFields(log.Fields{"HostId": id}).Debug("Host not found in database")
		}

		c.JSON(http.StatusOK, host)
	}
}

func (server *EnvServer) handleGetHostsRequest(c *gin.Context) {
	if hosts, err := server.db.GetHosts(); err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to get hosts")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		var err error
		for _, host := range hosts {
			host.VMs, err = server.calcNumberOfVMsPerHost(host.HostID)
			if err != nil {
				log.WithFields(log.Fields{"Error": err}).Error("Failed to get hosts")
				c.String(http.StatusInternalServerError, err.Error())
				return
			}
		}
		log.Debug("Getting hosts from database")
		c.JSON(http.StatusOK, hosts)
	}
}

func (server *EnvServer) handleRemoveHostRequest(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		log.WithFields(log.Fields{"Error": "Paramater id must be specified"}).Error("Failed to remove host")
		c.String(http.StatusBadRequest, "Paramater id must be specified")
		return
	}

	if err := server.db.RemoveHost(id); err != nil {
		log.WithFields(log.Fields{"Error": err}).Error("Failed to remove host")
		c.String(http.StatusInternalServerError, err.Error())
	} else {
		log.WithFields(log.Fields{"HostID": id}).Debug("Removing host from database")
		c.String(http.StatusOK, "")
	}
}
