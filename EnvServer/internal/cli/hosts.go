package cli

import (
	"os"
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/client"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

func init() {
	hostsCmd.AddCommand(addHostCmd)
	hostsCmd.AddCommand(getHostsCmd)
	hostsCmd.AddCommand(reportHostMetricCmd)
	rootCmd.AddCommand(hostsCmd)

	addHostCmd.Flags().StringVarP(&HostID, "hostid", "", "", "Host Id")
	addHostCmd.MarkFlagRequired("hostid")

	addHostCmd.Flags().StringVarP(&TotalCPU, "totalcpu", "", "", "Total CPU")
	addHostCmd.MarkFlagRequired("totalcpu")
	addHostCmd.Flags().StringVarP(&TotalMemory, "totalmem", "", "", "Total memory in bytes")
	addHostCmd.MarkFlagRequired("totalmem")

	reportHostMetricCmd.Flags().StringVarP(&HostID, "hostid", "", "", "Host Id")
	reportHostMetricCmd.MarkFlagRequired("hostid")
	reportHostMetricCmd.Flags().StringVarP(&UsageCPU, "cpu", "", "", "CPU usage")
	reportHostMetricCmd.MarkFlagRequired("cpu")
	reportHostMetricCmd.Flags().StringVarP(&UsageMemory, "mem", "", "", "Memory usage in bytes")
	reportHostMetricCmd.MarkFlagRequired("mem")
}

var hostsCmd = &cobra.Command{
	Use:   "hosts",
	Short: "Manage hosts",
	Long:  "Manage hosts",
}

var addHostCmd = &cobra.Command{
	Use:   "add",
	Short: "Add a new host",
	Long:  "Add a new host",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		if HostID == "" {
			log.Fatal("Host Id is required")
		}

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		totalCPU, err := strconv.ParseFloat(TotalCPU, 64)
		CheckError(err)

		totalMem, err := strconv.ParseFloat(TotalMemory, 64)
		CheckError(err)

		err = client.AddHost(&core.Host{HostID: HostID, TotalCPU: totalCPU, TotalMemory: totalMem})
		CheckError(err)

		log.WithFields(log.Fields{
			"HostId":      HostID,
			"TotalCPU":    TotalCPU,
			"TotalMemory": TotalMemory}).
			Info("Host added")
	},
}

var getHostsCmd = &cobra.Command{
	Use:   "ls",
	Short: "List all hosts",
	Long:  "List all hosts",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		hosts, err := client.GetHosts()
		CheckError(err)

		if len(hosts) == 0 {
			log.Info("No hosts found")
			os.Exit(0)
		}

		printHostsTable(hosts)
	},
}

var reportHostMetricCmd = &cobra.Command{
	Use:   "report",
	Short: "Report host metrics",
	Long:  "Report host metrics",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		if HostID == "" {
			log.Fatal("Host Id is required")
		}

		usageCPU, err := strconv.ParseFloat(UsageCPU, 64)
		CheckError(err)

		usageMem, err := strconv.ParseFloat(UsageMemory, 64)
		CheckError(err)

		metric := &core.Metric{Timestamp: time.Now(), Memory: usageMem, CPU: usageCPU}

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		err = client.AddMetric(HostID, core.HostType, metric)
		CheckError(err)

		log.WithFields(log.Fields{
			"HostId":    HostID,
			"UsageCPU":  UsageCPU,
			"UsageMem":  UsageMemory,
			"Timestamp": metric.Timestamp}).Info("Host metric reported")
	},
}
