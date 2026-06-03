// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"fmt"
	"strconv"
)

// This file holds the pure model-metadata helpers the HF download panel uses to
// size a model and render its parameter count from the safetensors metadata a
// repo advertises. The network fetch of that metadata is the caller's seam.

// dtypeBytes is the byte width of each safetensors dtype, used to turn a
// per-dtype parameter count into an on-disk byte size. Unknown dtypes count as
// one byte, matching the reference's default.
var dtypeBytes = map[string]int{
	"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
	"I64": 8, "I32": 4, "I16": 2, "I8": 1,
	"U64": 8, "U32": 4, "U16": 2, "U8": 1,
	"BOOL": 1,
}

// CalcSafetensorsDiskSize derives the on-disk byte size from a safetensors
// parameter map (dtype to element count), since the advertised total is a
// parameter count rather than bytes. Each dtype's count is scaled by its byte
// width, with an unknown dtype counting as one byte.
func CalcSafetensorsDiskSize(params map[string]int) int {
	total := 0
	for dtype, count := range params {
		width, ok := dtypeBytes[dtype]
		if !ok {
			width = 1
		}
		total += count * width
	}
	return total
}

// GetParamCount sums a safetensors parameter map into the model's total
// parameter count.
func GetParamCount(params map[string]int) int {
	total := 0
	for _, count := range params {
		total += count
	}
	return total
}

// FormatModelSize renders a byte size as a human-readable string, choosing KB,
// MB, or GB by magnitude and showing one decimal, matching the reference's
// thresholds and rounding.
func FormatModelSize(sizeBytes int) string {
	switch {
	case sizeBytes < 1024*1024:
		return fmt.Sprintf("%.1f KB", float64(sizeBytes)/1024)
	case sizeBytes < 1024*1024*1024:
		return fmt.Sprintf("%.1f MB", float64(sizeBytes)/(1024*1024))
	default:
		return fmt.Sprintf("%.1f GB", float64(sizeBytes)/(1024*1024*1024))
	}
}

// FormatParamCount renders a parameter count as a human-readable string with a
// T, B, or M suffix and one decimal, falling back to the bare integer below one
// million.
func FormatParamCount(totalParams int) string {
	p := float64(totalParams)
	switch {
	case p >= 1e12:
		return fmt.Sprintf("%.1fT", p/1e12)
	case p >= 1e9:
		return fmt.Sprintf("%.1fB", p/1e9)
	case p >= 1e6:
		return fmt.Sprintf("%.1fM", p/1e6)
	default:
		return strconv.Itoa(totalParams)
	}
}
