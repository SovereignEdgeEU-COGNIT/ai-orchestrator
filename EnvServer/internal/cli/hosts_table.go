package cli

import (
	"strconv"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/internal/table"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/pkg/core"
	"github.com/muesli/termenv"
)

func printHostsTable(hosts []*core.Host) {
	t, theme := createTable(0)

	var cols = []table.Column{
		{ID: "HostID", Name: "HostId", SortIndex: 1},
		{ID: "StateID", Name: "StateId", SortIndex: 2},
		{ID: "TotalCPU", Name: "TotalCPU", SortIndex: 3},
		{ID: "TotalMem", Name: "TotalMem", SortIndex: 4},
		{ID: "UsageCPU", Name: "UsageCPU", SortIndex: 5},
		{ID: "UsageMem", Name: "UsageMem", SortIndex: 6},
		{ID: "DiskRead", Name: "DiskRead", SortIndex: 7},
		{ID: "DiskWrite", Name: "DiskWrite", SortIndex: 8},
		{ID: "NetTX", Name: "NetTX", SortIndex: 9},
		{ID: "NetRX", Name: "NetRX", SortIndex: 10},
		{ID: "VMS", Name: "VMs", SortIndex: 11},
	}
	t.SetCols(cols)

	for _, host := range hosts {
		row := []interface{}{
			termenv.String(host.HostID).Foreground(theme.ColorCyan),
			termenv.String(strconv.Itoa(host.StateID)).Foreground(theme.ColorViolet),
			termenv.String(strconv.FormatFloat(host.TotalCPU, 'f', 2, 64)).Foreground(theme.ColorMagenta),
			termenv.String(strconv.FormatFloat(host.TotalMemory, 'f', 2, 64)).Foreground(theme.ColorMagenta),
			termenv.String(strconv.FormatFloat(host.UsageCPU, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.UsageMemory, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.DiskRead, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.DiskRead, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.DiskWrite, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.NetRX, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(host.NetTX, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.Itoa(host.VMs)).Foreground(theme.ColorBlue),
		}
		t.AddRow(row)
	}

	t.Render()
}
