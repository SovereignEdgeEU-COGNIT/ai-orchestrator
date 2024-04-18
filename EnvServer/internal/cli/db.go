package cli

import (
	"bufio"
	"fmt"
	"os"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/database"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

func init() {
	dbCmd.AddCommand(dbCreateCmd)
	dbCmd.AddCommand(dbDropCmd)
	dbCmd.AddCommand(exportCmd)
	rootCmd.AddCommand(dbCmd)

	dbCmd.PersistentFlags().StringVarP(&DBHost, "dbhost", "", DefaultDBHost, "DB host")
	dbCmd.PersistentFlags().IntVarP(&DBPort, "dbport", "", DefaultDBPort, "DB port")
	dbCmd.PersistentFlags().StringVarP(&DBUser, "dbuser", "", "", "DB user")
	dbCmd.PersistentFlags().StringVarP(&DBPassword, "dbpassword", "", "", "DB password")

	exportCmd.PersistentFlags().StringVarP(&MetricType, "type", "", "", "Metric type, use vm or host")
	exportCmd.PersistentFlags().StringVarP(&Filename, "filename", "", "", "Filename to export to")
	exportCmd.PersistentFlags().StringVarP(&ID, "id", "", "", "ID of the VM or host to export")
}

var dbCmd = &cobra.Command{
	Use:   "database",
	Short: "Manage internal database",
	Long:  "Manage internal database",
}

var dbCreateCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a database",
	Long:  "Create a database",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()
		parseDBEnv()
		parseMLClientEnv()

		var db *database.Database
		for {
			db = database.CreateDatabase(DBHost, DBPort, DBUser, DBPassword, DBName, DBPrefix)
			err := db.Connect()
			if err != nil {
				log.WithFields(log.Fields{"Error": err}).Error("Failed to call db.Connect(), retrying in 1 second ...")
				time.Sleep(1 * time.Second)
			} else {
				break
			}
		}

		log.WithFields(log.Fields{"Host": DBHost, "Port": DBPort, "User": DBUser, "Password": "**********************", "Prefix": DBPrefix}).Info("Connected to TimescaleDB")

		err := db.Initialize()
		if err != nil {
			log.WithFields(log.Fields{"Error": err}).Error("Failed to call db.Initialize()")
			os.Exit(0)
		}

		log.Info("TimescaleDB initialized")
	},
}

var dbDropCmd = &cobra.Command{
	Use:   "drop",
	Short: "Drop the database",
	Long:  "Drop the database",
	Run: func(cmd *cobra.Command, args []string) {
		parseDBEnv()
		parseMLClientEnv()

		fmt.Print("WARNING!!! Are you sure you want to drop the database? This operation cannot be undone! (YES,no): ")

		reader := bufio.NewReader(os.Stdin)
		reply, _ := reader.ReadString('\n')

		if reply == "YES\n" {
			log.WithFields(log.Fields{"DBHost": DBHost, "DBPort": DBPort, "DBUser": DBUser, "DBPassword": "*******************", "DBName": DBName, "UseTLS": UseTLS}).Info("Connecting to TimescaleDB")

			db := database.CreateDatabase(DBHost, DBPort, DBUser, DBPassword, DBName, DBPrefix)
			err := db.Connect()
			CheckError(err)

			err = db.Drop()
			CheckError(err)
			log.Info("TimescaleDB tables dropped")
		} else {
			log.Info("Aborting ...")
		}
	},
}

var exportCmd = &cobra.Command{
	Use:   "export",
	Short: "Export to CSV",
	Long:  "Export to CSV",
	Run: func(cmd *cobra.Command, args []string) {
		parseDBEnv()
		parseMLClientEnv()

		if Filename == "" {
			CheckError(fmt.Errorf("Filename is required"))
		}

		if MetricType == "" {
			CheckError(fmt.Errorf("Metric type is required"))
		}

		if ID == "" {
			CheckError(fmt.Errorf("ID is required"))
		}

		t := 0
		if MetricType == "vm" {
			t = core.VMType
		} else if MetricType == "host" {
			t = core.HostType
		} else {
			CheckError(fmt.Errorf("Invalid metric type"))
		}

		db := database.CreateDatabase(DBHost, DBPort, DBUser, DBPassword, DBName, DBPrefix)
		err := db.Connect()
		CheckError(err)

		err = db.Export(ID, t, Filename)
		CheckError(err)
	},
}
