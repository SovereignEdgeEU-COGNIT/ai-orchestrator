package cli

import (
	"os"
	"strconv"
	"time"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/client"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

func init() {
	vmsCmd.AddCommand(addVMCmd)
	vmsCmd.AddCommand(getVMsCmd)
	vmsCmd.AddCommand(bindCmd)
	vmsCmd.AddCommand(reportVMMetricCmd)
	rootCmd.AddCommand(vmsCmd)

	addVMCmd.Flags().StringVarP(&VMID, "vmid", "", "", "VM Id")
	addVMCmd.MarkFlagRequired("vmid")

	addVMCmd.Flags().StringVarP(&TotalCPU, "totalcpu", "", "", "Total CPU.")
	addVMCmd.MarkFlagRequired("totalcpu")
	addVMCmd.Flags().StringVarP(&TotalMemory, "totalmem", "", "", "Total memory in bytes")
	addVMCmd.MarkFlagRequired("totalmem")

	bindCmd.Flags().StringVarP(&VMID, "vmid", "", "", "VM Id")
	addVMCmd.MarkFlagRequired("vmid")

	bindCmd.Flags().StringVarP(&HostID, "hostid", "", "", "Host Id")
	addVMCmd.MarkFlagRequired("hostid")

	reportVMMetricCmd.Flags().StringVarP(&VMID, "vmid", "", "", "VM Id")
	reportVMMetricCmd.MarkFlagRequired("vmid")
	reportVMMetricCmd.Flags().StringVarP(&UsageCPU, "cpu", "", "", "CPU usage")
	reportVMMetricCmd.MarkFlagRequired("cpu")
	reportVMMetricCmd.Flags().StringVarP(&UsageMemory, "mem", "", "", "Memory usage in bytes")
	reportVMMetricCmd.MarkFlagRequired("mem")
}

var vmsCmd = &cobra.Command{
	Use:   "vms",
	Short: "Manage VMs",
	Long:  "Manage VMs",
}

var addVMCmd = &cobra.Command{
	Use:   "add",
	Short: "Add a new VM",
	Long:  "Add a new VM",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		if VMID == "" {
			log.Fatal("VM Id is required")
		}

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		totalCPU, err := strconv.ParseFloat(TotalCPU, 64)
		CheckError(err)

		totalMem, err := strconv.ParseFloat(TotalMemory, 64)
		CheckError(err)

		err = client.AddVM(&core.VM{VMID: VMID, TotalCPU: totalCPU, TotalMemory: totalMem})
		CheckError(err)

		log.WithFields(log.Fields{
			"VMId":        VMID,
			"TotalCPU":    TotalCPU,
			"TotalMemory": TotalMemory}).
			Info("VM added")
	},
}

var getVMsCmd = &cobra.Command{
	Use:   "ls",
	Short: "List all VMs",
	Long:  "List all VMs",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		vms, err := client.GetVMs()
		CheckError(err)

		if len(vms) == 0 {
			log.Info("No VM found")
			os.Exit(0)
		}

		printVMsTable(vms)
	},
}

var bindCmd = &cobra.Command{
	Use:   "bind",
	Short: "Bind a VM to a host",
	Long:  "Bind a VM to a host",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		if VMID == "" {
			log.Fatal("VM Id is required")
		}

		if HostID == "" {
			log.Fatal("Host Id is required")
		}

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		err := client.Bind(VMID, HostID)
		CheckError(err)

		log.WithFields(log.Fields{
			"VMId":   VMID,
			"HostId": HostID}).Info("VM bound")
	},
}

var reportVMMetricCmd = &cobra.Command{
	Use:   "report",
	Short: "Report VM metrics",
	Long:  "Report VM metrics",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		if VMID == "" {
			log.Fatal("VM Id is required")
		}

		usageCPU, err := strconv.ParseFloat(UsageCPU, 64)
		CheckError(err)

		usageMem, err := strconv.ParseFloat(UsageMemory, 64)
		CheckError(err)

		metric := &core.Metric{Timestamp: time.Now(), Memory: usageMem, CPU: usageCPU}

		client := client.CreateEnvClient(ServerHost, ServerPort, Insecure)

		err = client.AddMetric(VMID, core.VMType, metric)
		CheckError(err)

		log.WithFields(log.Fields{
			"VMId":      VMID,
			"UsageCPU":  UsageCPU,
			"UsageMem":  UsageMemory,
			"Timestamp": metric.Timestamp}).Info("VM metric reported")
	},
}
