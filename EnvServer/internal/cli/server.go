package cli

import (
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/database"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/opennebula"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/server"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Manage Env server",
	Long:  "Manage Env server",
}

func init() {
	serverCmd.AddCommand(serverStartCmd)
	rootCmd.AddCommand(serverCmd)

	serverCmd.PersistentFlags().BoolVarP(&InitDB, "initdb", "", false, "Initialize DB")
	serverCmd.PersistentFlags().BoolVarP(&One, "one", "", false, "Enable Opennebula integration")
	serverCmd.PersistentFlags().IntVarP(&ServerPort, "port", "", -1, "Server HTTP port")
}

var serverStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start a Env server",
	Long:  "Start a Env server",
	Run: func(cmd *cobra.Command, args []string) {
		parseDBEnv()
		parseEnv()
		parseMLClientEnv()

		var db *database.Database
		for {
			db = database.CreateDatabase(DBHost, DBPort, DBUser, DBPassword, DBName, DBPrefix)
			err := db.Connect()
			if err != nil {
				log.WithFields(log.Fields{"Error": err}).Error("Failed to connect to PostgreSQL database")
				time.Sleep(1 * time.Second)
			} else {
				break
			}
		}

		log.WithFields(log.Fields{"DBHost": DBHost, "DBPort": DBPort, "DBUser": DBUser, "DBPassword": "*******************", "DBName": DBName}).Info("Connected to PostgreSQL database")

		server := server.CreateEnvServer(db, ServerPort)

		if InitDB {
			err := db.Initialize()
			log.WithFields(log.Fields{"Error": err}).Info("Initialized DB")
		}

		go func() {
			for {
				err := server.ServeForever()
				if err != nil {
					log.WithFields(log.Fields{"Error": err}).Error("Failed to start Env server")
					time.Sleep(1 * time.Second)
				}
			}
		}()

		log.WithFields(log.Fields{"Port": ServerPort}).Info("Started Env server")
		if One {
			log.WithFields(log.Fields{"Port": 4567}).Info("Started OpenNebula ntegration server")
			prometheusURL := "http://" + PrometheusHost + ":" + strconv.Itoa(PrometheusPort)
			integrationServer := opennebula.CreateIntegrationServer(4567, prometheusURL, MLHost, MLPort, MLInsecure)

			go func() {
				for {
					err := integrationServer.ServeForever()
					if err != nil {
						log.WithFields(log.Fields{"Error": err}).Error("Failed to start OpenNebula Integration server")
						time.Sleep(1 * time.Second)
					}
				}
			}()

		}

		select {}
	},
}
