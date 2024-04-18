package cli

import (
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/emulator"
	"github.com/spf13/cobra"
)

var connectorsCmd = &cobra.Command{
	Use:   "connectors",
	Short: "Manage connector",
	Long:  "Manage connector",
}

var emulatorCmd = &cobra.Command{
	Use:   "emulator",
	Short: "Manage emulator connector",
	Long:  "Manage emulator connector",
}

func init() {
	emulatorCmd.AddCommand(startConnectorCmd)
	connectorsCmd.AddCommand(emulatorCmd)
	rootCmd.AddCommand(connectorsCmd)
}

var startConnectorCmd = &cobra.Command{
	Use:   "start",
	Short: "Start a connector",
	Long:  "Start a connector",
	Run: func(cmd *cobra.Command, args []string) {
		parseEnv()

		connector := emulator.CreateEmulatorConnector(CtrlPlaneHost, CtrlPlanePort, ServerHost, ServerPort, PrometheusHost, PrometheusPort)
		connector.Start()
	},
}
