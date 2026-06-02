// SPDX-License-Identifier: MIT OR Apache-2.0

// Package download holds the GPU-free planning logic for the model downloaders:
// repository-id validation, endpoint normalization, the sort-field map for hub
// search, and the size/parameter-count arithmetic and formatting drawn from a
// safetensors index. The network transfer itself (the hub client, the resumable
// snapshot download, progress polling) is not part of this package.
package download

import (
	"fmt"
	"strconv"
	"strings"
)

// DtypeBytes is the byte width of each safetensors dtype. An unknown dtype is
// treated as one byte (matching the upstream default).
var DtypeBytes = map[string]int64{
	"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
	"I64": 8, "I32": 4, "I16": 2, "I8": 1,
	"U64": 8, "U32": 4, "U16": 2, "U8": 1,
	"BOOL": 1,
}

// SortMap maps a UI sort option to the hub API sort field. The parameter- and
// size-based options fetch by downloads and are re-sorted client-side, so they
// map to "downloads" here.
var SortMap = map[string]string{
	"trending":     "trendingScore",
	"downloads":    "downloads",
	"created":      "createdAt",
	"updated":      "lastModified",
	"most_params":  "downloads",
	"least_params": "downloads",
	"largest":      "downloads",
	"smallest":     "downloads",
}

// SortField returns the hub API sort field for a UI sort option. The second
// result is false for an unknown option.
func SortField(sort string) (string, bool) {
	f, ok := SortMap[sort]
	return f, ok
}

// SafetensorsDiskSize computes the on-disk byte size of a model from its
// per-dtype parameter counts (a safetensors index reports parameter counts, not
// bytes). An unknown dtype counts as one byte per parameter. A nil/empty map is
// zero bytes.
func SafetensorsDiskSize(params map[string]int64) int64 {
	if len(params) == 0 {
		return 0
	}
	var total int64
	for dtype, count := range params {
		w, ok := DtypeBytes[dtype]
		if !ok {
			w = 1
		}
		total += count * w
	}
	return total
}

// ParamCount sums the per-dtype parameter counts into a total. A nil/empty map
// is zero.
func ParamCount(params map[string]int64) int64 {
	var total int64
	for _, count := range params {
		total += count
	}
	return total
}

// FormatModelSize renders a byte size as KB, MB, or GB with one decimal. Sizes
// under a mebibyte read as KB, so even tiny models show a KB figure.
func FormatModelSize(sizeBytes int64) string {
	const mib = 1024.0 * 1024.0
	const gib = mib * 1024.0
	b := float64(sizeBytes)
	switch {
	case b < mib:
		return oneDecimal(b/1024.0) + " KB"
	case b < gib:
		return oneDecimal(b/mib) + " MB"
	default:
		return oneDecimal(b/gib) + " GB"
	}
}

// FormatParamCount renders a parameter count as T, B, or M with one decimal, or
// the plain integer below a million.
func FormatParamCount(total int64) string {
	t := float64(total)
	switch {
	case t >= 1e12:
		return oneDecimal(t/1e12) + "T"
	case t >= 1e9:
		return oneDecimal(t/1e9) + "B"
	case t >= 1e6:
		return oneDecimal(t/1e6) + "M"
	default:
		return strconv.FormatInt(total, 10)
	}
}

// oneDecimal formats a float with one decimal place, matching Python's "%.1f"
// (round half to even, which Go's float formatter also uses).
func oneDecimal(x float64) string {
	return strconv.FormatFloat(x, 'f', 1, 64)
}

// NormalizeEndpoint strips trailing slashes from a hub endpoint URL.
func NormalizeEndpoint(endpoint string) string {
	return strings.TrimRight(endpoint, "/")
}

// ValidateRepoID trims a repository id and checks it has the owner/model shape
// (exactly one slash, two segments). It returns the trimmed id when valid, or an
// error whose message quotes the trimmed id.
func ValidateRepoID(repoID string) (string, error) {
	repoID = strings.TrimSpace(repoID)
	if !strings.Contains(repoID, "/") || len(strings.Split(repoID, "/")) != 2 {
		return "", fmt.Errorf("Invalid repository ID: '%s'. "+
			"Expected format: 'owner/model' (e.g., 'mlx-community/Llama-3-8B-4bit')", repoID)
	}
	return repoID, nil
}
