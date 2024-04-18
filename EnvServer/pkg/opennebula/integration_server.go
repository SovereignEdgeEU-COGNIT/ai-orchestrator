package opennebula

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

type IntegrationServer struct {
	ginHandler *gin.Engine
	port       int
	httpServer *http.Server
	monitor    *monitor
	MLClient   *MLClient
}

func CreateIntegrationServer(port int, prometheusURL string, mlHost string, mlPort int, mlInsecure bool) *IntegrationServer {
	server := &IntegrationServer{}
	server.ginHandler = gin.Default()
	server.ginHandler.Use(cors.Default())

	httpServer := &http.Server{
		Addr:    ":" + strconv.Itoa(port),
		Handler: server.ginHandler,
	}

	server.MLClient = CreateMLClient(mlHost, mlPort, mlInsecure)

	server.httpServer = httpServer
	server.port = port

	log.WithFields(log.Fields{"Port": port}).Info("Starting IntegrationServer")

	server.setupRoutes()

	server.monitor = newMonitor(prometheusURL)
	server.monitor.runForever()

	return server
}

func (server *IntegrationServer) setupRoutes() {
	server.ginHandler.POST("/", server.handlePlacementRequest)
}

func (server *IntegrationServer) ServeForever() error {
	if err := server.httpServer.ListenAndServe(); err != nil && errors.Is(err, http.ErrServerClosed) {
		return err
	}

	return nil
}

func (server *IntegrationServer) Shutdown() {
	server.monitor.stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.httpServer.Shutdown(ctx); err != nil {
		log.WithFields(log.Fields{"Error": err}).Warning("IntegrationServer forced to shutdown")
	}
}
