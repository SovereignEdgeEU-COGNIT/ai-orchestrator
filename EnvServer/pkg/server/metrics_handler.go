package server

import (
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

func (server *EnvServer) handleAddMetricRequest(c *gin.Context) {
	id, ok := c.GetQuery("id")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater id must be specified"}).Error("Failed to get metric")
		c.String(http.StatusBadRequest, "Paramater id must be specified")
		return
	}

	metricTypeStr, ok := c.GetQuery("metrictype")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater metrictype must be specified"}).Error("Failed to get metric")
		c.String(http.StatusBadRequest, "Paramater metrictype must be specified")
		return
	}
	metricType, err := strconv.Atoi(metricTypeStr)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error(), "Paramater": metricTypeStr}).Error("Failed to get metric")
		c.String(http.StatusBadRequest, "Paramater metrictype must be an integer")
		return
	}

	if metricType == core.HostType {
		host, err := server.db.GetHost(id)
		if err != nil {
			log.WithFields(log.Fields{"Error": err.Error(), "HostId": id}).Error("Failed to get host")
			c.String(http.StatusBadRequest, err.Error())
			return
		}
		if host == nil {
			log.WithFields(log.Fields{"Error": "Host does not exist", "HostId": id}).Error("Failed to get host")
			c.String(http.StatusBadRequest, "Host with id <"+id+"> does not exist")
			return
		}
	} else if metricType == core.VMType {
		vm, err := server.db.GetVM(id)
		if err != nil {
			log.WithFields(log.Fields{"Error": err.Error(), "VMId": id}).Error("Failed to get VM")
			c.String(http.StatusBadRequest, err.Error())
			return
		}
		if vm == nil {
			log.WithFields(log.Fields{"Error": "VM does not exist", "VMId": id}).Error("Failed to get VM")
			c.String(http.StatusBadRequest, "VM with id <"+id+"> does not exist")
			return
		}
	} else {
		log.WithFields(log.Fields{"Error": "Invalid metric type", "MetricType": metricType}).Error("Failed to get metric")
		c.String(http.StatusBadRequest, "Invalid metric type")
		return
	}

	jsonData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to read request body")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	metric, err := core.ConvertJSONToMetric(string(jsonData))
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to convert JSON to metric")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	err = server.db.AddMetric(id, metricType, metric)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to add metric")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	switch metricType {
	case core.HostType:
		err = server.db.SetHostResources(id, metric.CPU, metric.Memory, metric.DiskRead, metric.DiskWrite, metric.NetRX, metric.NetTX, metric.EnergyUsage)
		if err != nil {
			log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to set host resources")
			c.String(http.StatusBadRequest, err.Error())
			return
		}
	case core.VMType:
		err = server.db.SetVMResources(id, metric.CPU, metric.Memory, metric.DiskRead, metric.DiskWrite, metric.NetRX, metric.NetTX)
		if err != nil {
			log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to set VM resources")
			c.String(http.StatusBadRequest, err.Error())
			return
		}
	default:
		log.WithFields(log.Fields{"Error": "Invalid metric type", "MetricType": metricType}).Error("Failed to get metric")
		c.String(http.StatusBadRequest, "Invalid metric type")
	}

	log.WithFields(log.Fields{"Metric": metric}).Debug("Added metric")
	c.String(http.StatusOK, "")
}

func (server *EnvServer) handleGetMetricsRequest(c *gin.Context) {
	hostID, ok := c.GetQuery("hostid")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater hostid must be specified"}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater hostid must be specified")
		return
	}

	metricTypeStr, ok := c.GetQuery("metrictype")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater metrictype must be specified"}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater metrictype must be specified")
		return
	}
	metricType, err := strconv.Atoi(metricTypeStr)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error(), "Paramater": metricTypeStr}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater metrictype must be an integer")
		return
	}

	nanoUnixTimeStr, ok := c.GetQuery("since")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater since must be specified"}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater hostid must be specified")
		return
	}

	nanoUnixTime, err := strconv.ParseInt(nanoUnixTimeStr, 10, 64)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error(), "Paramater": nanoUnixTimeStr}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater since must be an integer")
		return
	}

	seconds := nanoUnixTime / int64(time.Second)
	nanoseconds := nanoUnixTime % int64(time.Second)
	ts := time.Unix(seconds, nanoseconds)

	countStr, ok := c.GetQuery("count")
	if !ok {
		log.WithFields(log.Fields{"Error": "Paramater count must be specified"}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater count must be specified")
		return
	}
	count, err := strconv.Atoi(countStr)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error(), "Paramater": countStr}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, "Paramater count must be an integer")
		return
	}

	metrics, err := server.db.GetMetrics(hostID, metricType, ts, count)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to get metrics")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	metricsJSON, err := core.ConvertMetricArrayToJSON(metrics)
	if err != nil {
		log.WithFields(log.Fields{"Error": err.Error()}).Error("Failed to convert metrics to JSON")
		c.String(http.StatusBadRequest, err.Error())
		return
	}

	log.Debug("Getting metrics")

	c.String(http.StatusOK, metricsJSON)
}
