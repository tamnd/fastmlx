// SPDX-License-Identifier: MIT OR Apache-2.0

// Package admin holds the side-effect-free cores of the admin panel's
// server-side logic. This file covers the accuracy-benchmark orchestration's
// pure parts: validating a start request and projecting a finished benchmark
// result into the wire shape the dashboard renders. The run loop itself (model
// load/unload, the per-question generate calls, the SSE event stream, and the
// server-side queue) stays a seam around an engine pool.
package admin

import (
	"slices"
	"strconv"

	"github.com/tamnd/fastmlx/eval"
)

// ValidBenchmarks is the set of benchmark names a start request may name, the
// same sixteen the eval registry serves.
var ValidBenchmarks = []string{
	"mmlu", "mmlu_pro", "kmmlu", "cmmlu", "jmmlu",
	"hellaswag", "truthfulqa", "arc_challenge", "winogrande",
	"gsm8k", "mathqa", "humaneval", "mbpp", "livecodebench",
	"bbq", "safetybench",
}

var validBatchSizes = map[int]struct{}{1: {}, 2: {}, 4: {}, 8: {}, 16: {}, 32: {}}

// ValidBatchSize reports whether v is an accepted batch size. The benchmark
// runner batches questions through the engine, and only these powers of two up
// to 32 are offered in the dashboard.
func ValidBatchSize(v int) bool {
	_, ok := validBatchSizes[v]
	return ok
}

// ValidBenchmark reports whether name is one the runner knows.
func ValidBenchmark(name string) bool {
	return slices.Contains(ValidBenchmarks, name)
}

// ValidBenchmarkSet reports whether a name-to-sample-size request is runnable:
// it must name at least one benchmark, every name must be known, and every
// sample size must be non-negative (0 means the full dataset).
func ValidBenchmarkSet(set map[string]int) bool {
	if len(set) == 0 {
		return false
	}
	for name, size := range set {
		if !ValidBenchmark(name) {
			return false
		}
		if size < 0 {
			return false
		}
	}
	return true
}

// ResultData projects a finished benchmark result into the dashboard wire shape,
// rounding the displayed numbers the way the reference does (accuracy and each
// category score to four decimals, the run time to one, each question's time to
// three) and renaming the question fields to their shorter wire keys. Category
// scores are omitted entirely when the benchmark has no categories, matching the
// reference's conditional key.
func ResultData(result eval.BenchmarkResult, modelID string) map[string]any {
	questions := make([]map[string]any, len(result.QuestionResults))
	for i, qr := range result.QuestionResults {
		questions[i] = map[string]any{
			"id":           qr.QuestionID,
			"correct":      qr.Correct,
			"expected":     qr.Expected,
			"predicted":    qr.Predicted,
			"question":     qr.QuestionText,
			"raw_response": qr.RawResponse,
			"category":     qr.Category,
			"time_s":       pyRound(qr.TimeSeconds, 3),
		}
	}

	data := map[string]any{
		"model_id":         modelID,
		"benchmark":        result.BenchmarkName,
		"accuracy":         pyRound(result.Accuracy, 4),
		"thinking_used":    result.ThinkingUsed,
		"total":            result.TotalQuestions,
		"correct":          result.CorrectCount,
		"time_s":           pyRound(result.TimeSeconds, 1),
		"question_results": questions,
	}

	if len(result.CategoryScores) > 0 {
		scores := make(map[string]any, len(result.CategoryScores))
		for k, v := range result.CategoryScores {
			scores[k] = pyRound(v, 4)
		}
		data["category_scores"] = scores
	}

	return data
}

// pyRound rounds x to ndigits decimal places the way CPython's round() does:
// correctly-rounded on the true binary value with ties going to the even digit.
// Go's strconv formats with exactly that rule, so formatting to the requested
// precision and parsing back reproduces round()'s result.
func pyRound(x float64, ndigits int) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', ndigits, 64), 64)
	return v
}
