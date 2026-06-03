// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

// This file holds the pure tail of the active-models dashboard builder: the
// memory-pressure block, the memory used/max selection, the final payload
// assembly, and the empty payload returned when the engine pool is absent. The
// per-model loop that needs engine internals is a caller seam, so the assembled
// models list and the request totals arrive as inputs. The enforcer status and
// the pool status arrive as plain dicts.
//
// One subtlety carries through: the field reads use an is-not-None test, so an
// empty enforcer dict still reads its defaults, while the enabled flag uses
// truthiness, so an empty dict is not enabled. A Go nil map stands for Python's
// None and an empty non-nil map for {}, which keeps the two apart.

// getDefault returns d[key] when the dict is non-nil and the key is present,
// else the default, mirroring `d.get(key, def) if d is not None else def`.
func getDefault(d map[string]any, key string, def any) any {
	if d != nil {
		if v, ok := d[key]; ok {
			return v
		}
	}
	return def
}

// BuildMemoryPressure builds the memory-pressure block from an enforcer status
// dict. A nil dict (Python None) and an empty dict both yield the all-default
// block; enabled is true only when the dict is non-empty and its enabled flag
// is truthy.
func BuildMemoryPressure(enforcerStatus map[string]any) map[string]any {
	return map[string]any{
		"enabled":           pyTruthy(enforcerStatus) && pyTruthy(enforcerStatus["enabled"]),
		"current_bytes":     getDefault(enforcerStatus, "current_bytes", 0),
		"soft_bytes":        getDefault(enforcerStatus, "soft_bytes", 0),
		"hard_bytes":        getDefault(enforcerStatus, "hard_bytes", 0),
		"current_formatted": getDefault(enforcerStatus, "current_formatted", "0.0GB"),
		"soft_formatted":    getDefault(enforcerStatus, "soft_formatted", "0.0GB"),
		"hard_formatted":    getDefault(enforcerStatus, "hard_formatted", "0.0GB"),
		"pressure_level":    getDefault(enforcerStatus, "pressure_level", "ok"),
	}
}

// SelectModelMemory picks the memory used and max for the usage bar. When the
// enforcer is present and enabled the values track its current and ceiling
// bytes so the bar matches what drives eviction; otherwise they fall back to
// the pool status's current model memory and final ceiling.
func SelectModelMemory(enforcerStatus, status map[string]any) (any, any) {
	if enforcerStatus != nil && pyTruthy(enforcerStatus["enabled"]) {
		return getDefault(enforcerStatus, "current_bytes", 0),
			getDefault(enforcerStatus, "ceiling_bytes", 0)
	}
	return getDefault(status, "current_model_memory", 0),
		getDefault(status, "final_ceiling", 0)
}

// AssembleActiveModelsData assembles the populated payload from the seam-built
// models list, the pool and enforcer status dicts, and the request totals.
func AssembleActiveModelsData(models []any, status, enforcerStatus map[string]any, totalActive, totalWaiting int) map[string]any {
	used, mx := SelectModelMemory(enforcerStatus, status)
	return map[string]any{
		"models":                 models,
		"model_memory_used":      used,
		"model_memory_max":       mx,
		"memory_pressure":        BuildMemoryPressure(enforcerStatus),
		"total_active_requests":  totalActive,
		"total_waiting_requests": totalWaiting,
	}
}

// EmptyActiveModelsData is the payload returned when the engine pool is absent:
// no models, zero memory, and the all-default memory-pressure block.
func EmptyActiveModelsData() map[string]any {
	return map[string]any{
		"models":                 []any{},
		"model_memory_used":      0,
		"model_memory_max":       0,
		"memory_pressure":        BuildMemoryPressure(nil),
		"total_active_requests":  0,
		"total_waiting_requests": 0,
	}
}
