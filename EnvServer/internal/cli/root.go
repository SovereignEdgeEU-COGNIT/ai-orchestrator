package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

const DBPrefix = "PROD_"
const TimeLayout = "2006-01-02 15:04:05"
const DefaultDBHost = "localhost"
const DefaultDBPort = 5432
const DefaultServerHost = "localhost"

var DBName = "postgres"
var Verbose bool
var DBHost string
var DBPort int
var DBUser string
var DBPassword string
var MLHost string
var MLPort int
var MLInsecure bool
var Insecure bool
var SkipTLSVerify bool
var UseTLS bool
var TLSCert string
var TLSKey string
var ServerHost string
var ServerPort int
var InitDB bool
var HostID string
var ASCII = false
var TotalCPU string
var TotalMemory string
var UsageCPU string
var UsageMemory string
var VMID string
var CtrlPlaneHost string
var CtrlPlanePort int
var PrometheusHost string
var PrometheusPort int
var Filename string
var MetricType string
var ID string
var One bool

func init() {
	rootCmd.PersistentFlags().BoolVarP(&Verbose, "verbose", "v", false, "Verbose (debugging)")
	rootCmd.PersistentFlags().BoolVarP(&Insecure, "insecure", "", false, "Disable TLS and use HTTP")
	rootCmd.PersistentFlags().BoolVarP(&SkipTLSVerify, "skip-tls-verify", "", false, "Skip TLS certificate verification")
}

var rootCmd = &cobra.Command{
	Use:   "envcli",
	Short: "Cognit Environment CLI tool",
	Long:  "Cogenv Environment CLI tool",
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
