package cli

import (
	"io"
	"os"
	"strconv"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/build"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
)

func CheckError(err error) {
	if err != nil {
		log.WithFields(log.Fields{"BuildVersion": build.BuildVersion, "BuildTime": build.BuildTime}).Error(err.Error())
		os.Exit(-1)
	}
}

func parseDBEnv() {
	DBHostEnv := os.Getenv("ENVSERVER_DB_HOST")
	if DBHostEnv != "" {
		DBHost = DBHostEnv
	}

	var err error
	dbPortEnvStr := os.Getenv("ENVSERVER__DB_PORT")
	if dbPortEnvStr != "" {
		DBPort, err = strconv.Atoi(dbPortEnvStr)
		CheckError(err)
	}

	if DBUser == "" {
		DBUser = os.Getenv("ENVSERVER_DB_USER")
	}

	if DBPassword == "" {
		DBPassword = os.Getenv("ENVSERVER_DB_PASSWORD")
	}
}

func parseMLClientEnv() {
	mlHost := os.Getenv("ML_HOST")
	if mlHost != "" {
		MLHost = mlHost
	}

	var err error
	mlPortEnvStr := os.Getenv("ML_PORT")
	if mlPortEnvStr != "" {
		MLPort, err = strconv.Atoi(mlPortEnvStr)
		CheckError(err)
	}

	mlInsecureStr := os.Getenv("ML_INSECURE")

	if mlInsecureStr == "true" {
		MLInsecure = false
	} else {
		MLInsecure = true
	}

}

func parseEnv() {
	var err error

	ServerHostEnv := os.Getenv("ENVSERVER_HOST")
	if ServerHostEnv != "" {
		ServerHost = ServerHostEnv
	}

	ServerPortEnvStr := os.Getenv("ENVSERVER_PORT")
	if ServerPortEnvStr != "" {
		if ServerPort == -1 {
			ServerPort, err = strconv.Atoi(ServerPortEnvStr)
			if err != nil {
				log.Error("Failed to parse ENVSERVER_PORT")
			}
			CheckError(err)
		}
	}

	if !Verbose {
		VerboseEnv := os.Getenv("ENVSERVER_VERBOSE")
		if VerboseEnv == "true" {
			Verbose = true
		} else if VerboseEnv == "false" {
			Verbose = false
		}

		if Verbose {
			log.SetLevel(log.DebugLevel)
		} else {
			log.SetLevel(log.InfoLevel)
			gin.SetMode(gin.ReleaseMode)
			gin.DefaultWriter = io.Discard
		}
	}

	TLSEnv := os.Getenv("ENVSERVER_TLS")
	if TLSEnv == "true" {
		UseTLS = true
		Insecure = false
	} else if TLSEnv == "false" {
		UseTLS = false
		Insecure = true
	}

	CtrlPlaneHost = os.Getenv("CTRLPLANE_HOST")

	ctrlPlanePortStr := os.Getenv("CTRLPLANE_PORT")
	if ctrlPlanePortStr != "" {
		CtrlPlanePort, err = strconv.Atoi(ctrlPlanePortStr)
		CheckError(err)
	}

	PrometheusHost = os.Getenv("PROMETHEUS_HOST")

	prometheusPortStr := os.Getenv("PROMETHEUS_PORT")
	if prometheusPortStr != "" {
		PrometheusPort, err = strconv.Atoi(prometheusPortStr)
		CheckError(err)
	}
}
