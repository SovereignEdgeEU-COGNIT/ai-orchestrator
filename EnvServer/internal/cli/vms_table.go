package cli

import (
	"strconv"

	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/internal/table"
	"github.com/SovereignEdgeEU-COGNIT/ai-orchestrator/EnvServer/pkg/core"
	"github.com/muesli/termenv"
)

func printVMsTable(vms []*core.VM) {
	t, theme := createTable(0)

	var cols = []table.Column{
		{ID: "VMID", Name: "VMId", SortIndex: 1},
		{ID: "StateID", Name: "StateId", SortIndex: 2},
		{ID: "Deployed", Name: "Deployed", SortIndex: 3},
		{ID: "HostID", Name: "HostID", SortIndex: 4},
		{ID: "HostStateID", Name: "HostStateID", SortIndex: 5},
		{ID: "TotalCPU", Name: "TotalCPU", SortIndex: 6},
		{ID: "TotalMem", Name: "TotalMem", SortIndex: 7},
		{ID: "UsageCPU", Name: "UsageCPU", SortIndex: 8},
		{ID: "UsageMem", Name: "UsageMem", SortIndex: 9},
		{ID: "DiskRead", Name: "DiskRead", SortIndex: 10},
		{ID: "DiskWrite", Name: "DiskWrite", SortIndex: 11},
		{ID: "NetTX", Name: "NetTX", SortIndex: 12},
		{ID: "NetRX", Name: "NetRX", SortIndex: 13},
	}
	t.SetCols(cols)

	for _, vm := range vms {
		row := []interface{}{
			termenv.String(vm.VMID).Foreground(theme.ColorCyan),
			termenv.String(strconv.Itoa(vm.StateID)).Foreground(theme.ColorViolet),
			termenv.String(strconv.FormatBool(vm.Deployed)).Foreground(theme.ColorBlue),
			termenv.String(vm.HostID).Foreground(theme.ColorBlue),
			termenv.String(strconv.Itoa(vm.HostStateID)).Foreground(theme.ColorViolet),
			termenv.String(strconv.FormatFloat(vm.TotalCPU, 'f', 2, 64)).Foreground(theme.ColorMagenta),
			termenv.String(strconv.FormatFloat(vm.TotalMemory, 'f', 2, 64)).Foreground(theme.ColorMagenta),
			termenv.String(strconv.FormatFloat(vm.UsageCPU, 'f', 2, 64)).Foreground(theme.ColorGreen),
			termenv.String(strconv.FormatFloat(vm.UsageMemory, 'f', 2, 64)).Foreground(theme.ColorGreen),
			termenv.String(strconv.FormatFloat(vm.DiskRead, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(vm.DiskRead, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(vm.DiskWrite, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(vm.NetRX, 'f', 2, 64)).Foreground(theme.ColorYellow),
			termenv.String(strconv.FormatFloat(vm.NetTX, 'f', 2, 64)).Foreground(theme.ColorYellow),
		}
		t.AddRow(row)
	}

	t.Render()
}
