package main

import (
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/internal/cli"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/build"
)

var (
	BuildVersion string = ""
	BuildTime    string = ""
)

func main() {
	build.BuildVersion = BuildVersion
	build.BuildTime = BuildTime
	cli.Execute()
}
