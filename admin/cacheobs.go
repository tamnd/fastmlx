// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"sort"
	"strconv"
)

// This file holds the pure portion of the runtime-cache observability builder
// the status dashboard renders: the per-model entry builder and the payload
// aggregation. The engine-pool walk, the scheduler and dataclass extraction
// that yields each model's runtime stats, and the directory-scan fallback that
// fires when no loaded model contributes stats all stay caller seams, so the
// inputs here are the already-extracted runtime_stats dicts (with ssd_cache and
// prefix_cache already reduced to plain dicts and an "id" injected) plus the
// resolved path and config values.

// CacheObsConfig carries the resolved paths and the configured disk cap the
// builder folds into the payload. These come from the global settings, a seam.
type CacheObsConfig struct {
	BasePath         string
	SsdCacheDir      string
	ResponseStateDir string
	DiskMaxBytes     int
}

// asPyInt returns the int value of a Python-int value (int or bool), matching
// the value isinstance(x, int) gates on. A bool counts (bool is an int subclass)
// but a float does not, which is the distinction the block-size and indexed-
// blocks checks turn on.
func asPyInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case bool:
		if n {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

// pyDictIntOr reproduces int(d.get(key, 0) or 0): the value or 0 when absent,
// then 0 again when that value is falsy, then int() truncation toward zero.
func pyDictIntOr(d map[string]any, key string) int {
	v, ok := d[key]
	if !ok {
		v = 0
	}
	n, _ := numToInt(pyOr(v, 0))
	return n
}

// dictOrEmpty returns the value as a dict, or an empty dict when it is not one,
// mirroring the `elif not isinstance(x, dict): x = {}` normalization.
func dictOrEmpty(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

// BuildCacheModelEntry builds one model's observability entry from its extracted
// runtime stats. block_size falls back to the prefix-cache block size when it is
// not a positive int; indexed_blocks falls back to zero when it is not an int;
// the sub-block-cache flag fires when nothing is indexed yet but partial-block
// skips have happened at a known block size, and it switches the display to the
// "<block_size" form. A cache_rates value is attached only when truthy.
func BuildCacheModelEntry(modelID any, runtimeStats map[string]any) map[string]any {
	ssdStats := dictOrEmpty(runtimeStats["ssd_cache"])
	prefixStats := dictOrEmpty(runtimeStats["prefix_cache"])

	indexedBlocks := 0
	if iv, ok := asPyInt(runtimeStats["indexed_blocks"]); ok {
		indexedBlocks = iv
	}

	var blockSize int
	if iv, ok := asPyInt(runtimeStats["block_size"]); ok && iv > 0 {
		blockSize = iv
	} else {
		blockSize = pyDictIntOr(prefixStats, "block_size")
	}

	partialBlockSkips := pyDictIntOr(prefixStats, "partial_block_skips")
	partialTokensSkipped := pyDictIntOr(prefixStats, "partial_tokens_skipped")
	lastPartialTokensSkipped := pyDictIntOr(prefixStats, "last_partial_tokens_skipped")
	lastTokensToNextBlock := pyDictIntOr(prefixStats, "last_tokens_to_next_block")

	hasSubBlockCache := indexedBlocks == 0 && blockSize > 0 && partialBlockSkips > 0

	display := strconv.Itoa(indexedBlocks)
	if hasSubBlockCache {
		display = "<" + strconv.Itoa(blockSize)
	}

	entry := map[string]any{
		"id":                          modelID,
		"block_size":                  blockSize,
		"indexed_blocks":              indexedBlocks,
		"indexed_blocks_display":      display,
		"has_sub_block_cache":         hasSubBlockCache,
		"partial_block_skips":         partialBlockSkips,
		"partial_tokens_skipped":      partialTokensSkipped,
		"last_partial_tokens_skipped": lastPartialTokensSkipped,
		"last_tokens_to_next_block":   lastTokensToNextBlock,
		"num_files":                   pyDictIntOr(ssdStats, "num_files"),
		"total_size_bytes":            pyDictIntOr(ssdStats, "total_size_bytes"),
		"max_size_bytes":              pyDictIntOr(ssdStats, "max_size_bytes"),
		"hot_cache_max_bytes":         pyDictIntOr(ssdStats, "hot_cache_max_bytes"),
		"hot_cache_size_bytes":        pyDictIntOr(ssdStats, "hot_cache_size_bytes"),
		"hot_cache_entries":           pyDictIntOr(ssdStats, "hot_cache_entries"),
	}
	if cr := runtimeStats["cache_rates"]; pyTruthy(cr) {
		entry["cache_rates"] = cr
	}
	return entry
}

// EmptyCacheObservability is the payload returned when the global settings are
// absent: the resolved paths are blank and every count is zero. It is a smaller
// shape than the populated payload, carrying no disk or hot-cache caps.
func EmptyCacheObservability() map[string]any {
	return map[string]any{
		"base_path":             "",
		"ssd_cache_dir":         "",
		"response_state_dir":    "",
		"models":                []any{},
		"total_num_files":       0,
		"total_size_bytes":      0,
		"effective_block_sizes": []any{},
	}
}

// BuildCacheObservability assembles the populated payload from the resolved
// config and the loaded models' extracted runtime stats. Per-model entries are
// built and summed, the effective block sizes are the sorted distinct positive
// block sizes, the hot-cache caps sum across models since each reserves its own
// slice of one process-wide budget, and the disk cap takes the largest of the
// configured cap and the per-model caps because one cache directory is shared.
func BuildCacheObservability(cfg CacheObsConfig, models []map[string]any) map[string]any {
	entries := []any{}
	totalNumFiles := 0
	totalSizeBytes := 0
	blockSizeSet := map[int]struct{}{}

	hotCacheMax := 0
	hotCacheSize := 0
	hotCacheEntries := 0
	diskMax := cfg.DiskMaxBytes

	for _, rs := range models {
		entry := BuildCacheModelEntry(rs["id"], rs)
		entries = append(entries, entry)
		totalNumFiles += entry["num_files"].(int)
		totalSizeBytes += entry["total_size_bytes"].(int)
		if bs := entry["block_size"].(int); bs > 0 {
			blockSizeSet[bs] = struct{}{}
		}
		hotCacheSize += entry["hot_cache_size_bytes"].(int)
		hotCacheEntries += entry["hot_cache_entries"].(int)
		hotCacheMax += entry["hot_cache_max_bytes"].(int)
		diskMax = max(diskMax, entry["max_size_bytes"].(int))
	}

	effective := make([]int, 0, len(blockSizeSet))
	for bs := range blockSizeSet {
		effective = append(effective, bs)
	}
	sort.Ints(effective)
	effectiveAny := make([]any, len(effective))
	for i, bs := range effective {
		effectiveAny[i] = bs
	}

	return map[string]any{
		"base_path":             cfg.BasePath,
		"ssd_cache_dir":         cfg.SsdCacheDir,
		"response_state_dir":    cfg.ResponseStateDir,
		"models":                entries,
		"total_num_files":       totalNumFiles,
		"total_size_bytes":      totalSizeBytes,
		"effective_block_sizes": effectiveAny,
		"disk_max_bytes":        diskMax,
		"hot_cache_max_bytes":   hotCacheMax,
		"hot_cache_size_bytes":  hotCacheSize,
		"hot_cache_entries":     hotCacheEntries,
	}
}
