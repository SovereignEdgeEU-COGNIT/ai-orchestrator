package main

import (
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/internal/cli"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/build"
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
