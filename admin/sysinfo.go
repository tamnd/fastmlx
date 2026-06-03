// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

// This file holds the pure dict assemblers behind the admin dashboard's disk
// and system-memory panels. The actual measurements (disk_usage on the cache
// directory, the hw.memsize sysctl, psutil available memory, the Metal wired
// cap, the macOS vm_stat layers) all stay caller seams: the disk-info and
// memory-info builders take the already-measured byte counts and shape the JSON
// the dashboard reads, so the size formatting and the 80% auto-limit math are
// the testable cores.

// SsdDiskInfo shapes the cache-directory disk panel. When the disk_usage probe
// fails the caller passes ok=false and the panel shows a zero total with an
// "Unknown" formatted size; on success the total is formatted with the detailed
// B/KB/MB/GB/TB size string.
func SsdDiskInfo(totalBytes int, ok bool) map[string]any {
	if !ok {
		return map[string]any{"total_bytes": 0, "total_formatted": "Unknown"}
	}
	return map[string]any{
		"total_bytes":     totalBytes,
		"total_formatted": FormatSizeDetailed(totalBytes),
	}
}

// AutoMemoryLimit is the suggested memory ceiling the dashboard previews: 80% of
// physical RAM, truncated toward zero to match Python's int(total * 0.8).
func AutoMemoryLimit(totalBytes int) int {
	return int(float64(totalBytes) * 0.8)
}

// SystemMemoryInputs carries the already-measured byte counts the memory panel
// renders. TotalBytes is hw.memsize; AvailableBytes is psutil's available; the
// remaining fields are the live process footprint, the effective Metal wired
// cap and the value requested at startup, and the macOS vm_stat layers. Each is
// zero when its probe is unavailable on the host.
type SystemMemoryInputs struct {
	TotalBytes             int
	AvailableBytes         int
	PhysFootprintBytes     int
	IogpuWiredLimitBytes   int
	WiredLimitRequestBytes int
	FreeMemoryBytes        int
	InactiveMemoryBytes    int
	ActiveMemoryBytes      int
}

// SystemMemoryInfo shapes the system-memory panel. It derives the 80% auto
// limit, formats the total and the auto limit, and passes the measured layers
// straight through so the dashboard JS can preview the tier-aware ceiling.
func SystemMemoryInfo(in SystemMemoryInputs) map[string]any {
	autoLimit := AutoMemoryLimit(in.TotalBytes)
	return map[string]any{
		"total_bytes":                       in.TotalBytes,
		"total_formatted":                   FormatSizeDetailed(in.TotalBytes),
		"auto_limit_bytes":                  autoLimit,
		"auto_limit_formatted":              FormatSizeDetailed(autoLimit),
		"available_bytes":                   in.AvailableBytes,
		"fastmlx_phys_footprint_bytes":      in.PhysFootprintBytes,
		"iogpu_wired_limit_bytes":           in.IogpuWiredLimitBytes,
		"fastmlx_wired_limit_request_bytes": in.WiredLimitRequestBytes,
		"free_memory_bytes":                 in.FreeMemoryBytes,
		"inactive_memory_bytes":             in.InactiveMemoryBytes,
		"active_memory_bytes":               in.ActiveMemoryBytes,
	}
}
