// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// This file holds the pure cores of the throughput-benchmark admin panel: the
// per-request metrics math, the quantization-label detection, the display-name
// cleaning, and the start-request validation. The timed generation loop, the
// SSE stream, and the optional leaderboard upload stay seams in the caller.

// ValidPromptLengths is the set of prompt token counts the throughput panel
// offers, in display order.
var ValidPromptLengths = []int{1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000}

// ValidThroughputBatchSizes is the set of concurrent batch sizes the throughput
// panel offers. (The accuracy panel uses its own batch-size set.)
var ValidThroughputBatchSizes = []int{2, 4, 8}

var (
	quantSuffixRe = regexp.MustCompile(`(?i)[-_](2bit|3bit|4bit|6bit|8bit|fp16|bf16|fp32|MXFP4|NVFP4)$`)
	mlxSuffixRe   = regexp.MustCompile(`(?i)[-_]?MLX[-_]?`)
	nameQuantRe   = regexp.MustCompile(`(?i)(2bit|3bit|4bit|6bit|8bit|fp16|bf16|MXFP4|NVFP4)`)
)

// ComputeSingleMetrics derives the throughput figures for one timed request from
// its token counts and timestamps, rounding each for display the way the
// reference does and passing the raw token and memory counts through unchanged.
// The 1e-9 floors guard the divisions against a zero-length window, and tpot is
// spread across the tokens after the first since the first is the prefill cost.
func ComputeSingleMetrics(promptTokens, completionTokens int, startTime, firstTokenTime, endTime float64, peakMemory, cachedTokens int) map[string]any {
	ttftS := firstTokenTime - startTime
	genDuration := endTime - firstTokenTime
	e2eDuration := endTime - startTime

	ttftMs := ttftS * 1000
	tpotMs := (genDuration / float64(max(completionTokens-1, 1))) * 1000
	genTps := float64(completionTokens) / max(genDuration, 1e-9)
	processingTps := float64(promptTokens) / max(ttftS, 1e-9)
	totalThroughput := float64(promptTokens+completionTokens) / max(e2eDuration, 1e-9)

	return map[string]any{
		"ttft_ms":           pyRound(ttftMs, 1),
		"tpot_ms":           pyRound(tpotMs, 2),
		"gen_tps":           pyRound(genTps, 1),
		"processing_tps":    pyRound(processingTps, 1),
		"e2e_latency_s":     pyRound(e2eDuration, 3),
		"total_throughput":  pyRound(totalThroughput, 1),
		"peak_memory_bytes": peakMemory,
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"cached_tokens":     cachedTokens,
	}
}

// DetectQuantization resolves a model's quantization label. The config.json read
// is the caller's seam: when the parsed quantization_config carries a bit count
// the caller passes it (hasBits true) and this returns "<bits>bit"; otherwise the
// model's directory name is searched for a known marker, falling back to
// "unknown".
func DetectQuantization(configBits int, hasBits bool, dirName string) string {
	if hasBits {
		return strconv.Itoa(configBits) + "bit"
	}
	return DetectQuantizationFromName(dirName)
}

// DetectQuantizationFromName searches a directory name for a known quantization
// marker, returning it lowercased, or "unknown" when none is present.
func DetectQuantizationFromName(dirName string) string {
	if m := nameQuantRe.FindString(dirName); m != "" {
		return strings.ToLower(m)
	}
	return "unknown"
}

// CleanModelName strips the trailing quantization suffix and any MLX markers from
// a model id so it reads as a plain display name (e.g. "Qwen3-30B-A3B-4bit" ->
// "Qwen3-30B-A3B"), trimming leftover separators.
func CleanModelName(modelID string) string {
	name := quantSuffixRe.ReplaceAllString(modelID, "")
	name = mlxSuffixRe.ReplaceAllString(name, "")
	return strings.Trim(name, "-_ ")
}

// ValidatePromptLengths checks a requested set of prompt lengths: it must be
// non-empty and every entry must be an offered length. On success it returns the
// lengths sorted (the normalized form the run uses); otherwise ok is false.
func ValidatePromptLengths(v []int) ([]int, bool) {
	if len(v) == 0 {
		return nil, false
	}
	for _, pl := range v {
		if !slices.Contains(ValidPromptLengths, pl) {
			return nil, false
		}
	}
	out := append([]int{}, v...)
	slices.Sort(out)
	return out, true
}

// ValidateBatchSizes checks a requested set of batch sizes: every entry must be
// an offered size (an empty set is allowed, meaning single-request only). On
// success it returns the sizes sorted; otherwise ok is false.
func ValidateBatchSizes(v []int) ([]int, bool) {
	for _, bs := range v {
		if !slices.Contains(ValidThroughputBatchSizes, bs) {
			return nil, false
		}
	}
	out := append([]int{}, v...)
	slices.Sort(out)
	return out, true
}
