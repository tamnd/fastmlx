// SPDX-License-Identifier: MIT OR Apache-2.0

// Package bench holds the GPU-free math of the throughput benchmark: the
// per-request and per-batch latency/throughput metrics computed from timestamps
// and token counts, plus the model-name and quantization helpers used to label a
// run. Driving the engine and gathering the timestamps is not part of this
// package.
package bench

import (
	"math"
	"regexp"
	"strings"
)

// SingleInput is one request's measured timestamps and token counts.
type SingleInput struct {
	PromptTokens     int
	CompletionTokens int
	StartTime        float64
	FirstTokenTime   float64
	EndTime          float64
	PeakMemory       int64
	CachedTokens     int
}

// SingleMetrics is the derived per-request report.
type SingleMetrics struct {
	TTFTMs           float64
	TPOTMs           float64
	GenTPS           float64
	ProcessingTPS    float64
	E2ELatencyS      float64
	TotalThroughput  float64
	PeakMemoryBytes  int64
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
}

// ComputeSingleMetrics derives the latency and throughput figures for one
// request. Time-to-first-token is the prefill latency; time-per-output-token
// averages over the tokens after the first; generation and processing rates and
// the end-to-end throughput guard against a zero denominator. Each figure is
// rounded to the same number of decimals as upstream.
func ComputeSingleMetrics(in SingleInput) SingleMetrics {
	ttftS := in.FirstTokenTime - in.StartTime
	genDuration := in.EndTime - in.FirstTokenTime
	e2eDuration := in.EndTime - in.StartTime

	ttftMs := ttftS * 1000
	denom := max(in.CompletionTokens-1, 1)
	tpotMs := (genDuration / float64(denom)) * 1000
	genTPS := float64(in.CompletionTokens) / math.Max(genDuration, 1e-9)
	processingTPS := float64(in.PromptTokens) / math.Max(ttftS, 1e-9)
	totalThroughput := float64(in.PromptTokens+in.CompletionTokens) / math.Max(e2eDuration, 1e-9)

	return SingleMetrics{
		TTFTMs:           roundN(ttftMs, 1),
		TPOTMs:           roundN(tpotMs, 2),
		GenTPS:           roundN(genTPS, 1),
		ProcessingTPS:    roundN(processingTPS, 1),
		E2ELatencyS:      roundN(e2eDuration, 3),
		TotalThroughput:  roundN(totalThroughput, 1),
		PeakMemoryBytes:  in.PeakMemory,
		PromptTokens:     in.PromptTokens,
		CompletionTokens: in.CompletionTokens,
		CachedTokens:     in.CachedTokens,
	}
}

// BatchResult is one request's contribution to a batch run: its completion
// count, its time-to-first-token, and the absolute time it produced that token.
type BatchResult struct {
	CompletionTokens int
	TTFTS            float64
	FirstTokenAbs    float64
}

// BatchMetrics is the aggregate report for a batch run.
type BatchMetrics struct {
	PPTPS          float64
	TGTPS          float64
	AvgTTFTMs      float64
	E2ELatencyS    float64
	TotalGenTokens int
	BatchSize      int
}

// BatchAggregate combines a batch of request results. Prefill throughput (pp)
// counts all prompt tokens against the time until the last request finishes
// prefill; generation throughput (tg) counts all generated tokens against the
// wall time after that last prefill completes.
func BatchAggregate(results []BatchResult, promptTokens, batchSize int, wallStart, wallEnd float64) BatchMetrics {
	totalGen := 0
	sumTTFT := 0.0
	maxFirstToken := math.Inf(-1)
	for _, r := range results {
		totalGen += r.CompletionTokens
		sumTTFT += r.TTFTS
		if r.FirstTokenAbs > maxFirstToken {
			maxFirstToken = r.FirstTokenAbs
		}
	}
	totalPrompt := promptTokens * batchSize
	wallTime := wallEnd - wallStart
	avgTTFTMs := (sumTTFT / float64(batchSize)) * 1000

	prefillWall := maxFirstToken - wallStart
	ppTPS := float64(totalPrompt) / math.Max(prefillWall, 1e-9)
	genWall := wallEnd - maxFirstToken
	tgTPS := float64(totalGen) / math.Max(genWall, 1e-9)

	return BatchMetrics{
		PPTPS:          roundN(ppTPS, 1),
		TGTPS:          roundN(tgTPS, 1),
		AvgTTFTMs:      roundN(avgTTFTMs, 1),
		E2ELatencyS:    roundN(wallTime, 3),
		TotalGenTokens: totalGen,
		BatchSize:      batchSize,
	}
}

var (
	quantSuffixRe = regexp.MustCompile(`(?i)[-_](2bit|3bit|4bit|6bit|8bit|fp16|bf16|fp32|MXFP4|NVFP4)$`)
	mlxSuffixRe   = regexp.MustCompile(`(?i)[-_]?MLX[-_]?`)
	quantDetectRe = regexp.MustCompile(`(?i)(2bit|3bit|4bit|6bit|8bit|fp16|bf16|MXFP4|NVFP4)`)
)

// DetectQuantizationFromName reads a quantization label from a model directory
// name, lowercased, or "unknown" when none is present. (fp32 is intentionally
// not in the detection set, matching upstream.)
func DetectQuantizationFromName(dirname string) string {
	m := quantDetectRe.FindStringSubmatch(dirname)
	if m == nil {
		return "unknown"
	}
	return strings.ToLower(m[1])
}

// CleanModelName strips a trailing quantization suffix and any MLX markers from a
// model id for display, then trims leading/trailing separators.
func CleanModelName(modelID string) string {
	name := quantSuffixRe.ReplaceAllString(modelID, "")
	name = mlxSuffixRe.ReplaceAllString(name, "")
	return strings.Trim(name, "-_ ")
}

// roundN rounds to n decimal places, half to even, matching Python's round().
func roundN(x float64, n int) float64 {
	p := math.Pow(10, float64(n))
	return math.RoundToEven(x*p) / p
}
