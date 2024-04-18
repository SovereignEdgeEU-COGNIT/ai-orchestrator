package server

import (
	"io"
	"testing"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/client"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/database"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
)

const TESTHOST = "localhost"
const TESTPORT = 30090

func prepareTests(t *testing.T) (*client.EnvClient, *EnvServer, chan bool) {
	client := client.CreateEnvClient(TESTHOST, TESTPORT, true)
	gin.SetMode(gin.ReleaseMode)
	log.SetLevel(log.PanicLevel)
	gin.DefaultWriter = io.Discard

	db, err := database.PrepareTests()
	assert.Nil(t, err)

	server := CreateEnvServer(db, TESTPORT)
	assert.Nil(t, err)

	done := make(chan bool)

	go func() {
		server.ServeForever()
		done <- true
	}()

	time.Sleep(1 * time.Second) //Lazy method of waiting for server to start

	log.SetReportCaller(true)

	return client, server, done
}
