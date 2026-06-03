// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"math"
	"sort"
)

// Sampler selection math mirrors the compile-free mlx-lm sampler used by the
// serving layer (the apply_top_p / apply_min_p / apply_top_k filters and the
// make_sampler composition). The filters are pure: each masks tokens out of a
// logits row by setting them to -inf, which is plain arithmetic over a
// []float32 row, so they run and are parity-tested on any host. Greedy
// selection (temperature 0) is argmax, also pure.
//
// The MLX/RNG seam is the final categorical draw: make_sampler finishes with
// mx.random.categorical(logits * (1/temp)), which advances mlx's RNG state and
// is not reproducible in pure Go without reimplementing that generator. That
// draw, and the optional XTC step (which needs mx.random.uniform), stay in the
// cgo path. Everything here is the deterministic filter cascade that runs
// before it.

// negInf is the mask value, matching mlx's -float("inf").
var negInf = float32(math.Inf(-1))

// Sampler holds the sampling parameters, mirroring make_sampler's signature.
// Temp == 0 means greedy (argmax). The filters are applied in the same order as
// make_sampler: top-p, then min-p, then top-k.
type Sampler struct {
	Temp            float64
	TopP            float64
	MinP            float64
	TopK            int
	MinTokensToKeep int
}

// Greedy reports whether sampling reduces to argmax (temperature 0).
func (s Sampler) Greedy() bool { return s.Temp == 0 }

// ApplyFilters runs the enabled filter cascade over the logits row in place,
// matching make_sampler's composition: top-p is applied when 0 < top_p < 1,
// min-p when min_p != 0, and top-k when 0 < top_k < vocab. The remaining
// categorical draw is the MLX seam. When Greedy() is true the row is left
// untouched (the caller takes Argmax instead).
func (s Sampler) ApplyFilters(logits []float32) {
	if s.Greedy() {
		return
	}
	if s.TopP > 0 && s.TopP < 1.0 {
		ApplyTopP(logits, s.TopP)
	}
	if s.MinP != 0.0 {
		ApplyMinP(logits, s.MinP, s.MinTokensToKeep)
	}
	if s.TopK > 0 && s.TopK < len(logits) {
		ApplyTopK(logits, s.TopK)
	}
}

// Argmax returns the index of the highest logit, ties resolved to the first
// occurrence (matching mx.argmax). Returns -1 on an empty row.
func Argmax(logits []float32) int {
	best := -1
	var bestVal float32
	for i, v := range logits {
		if best == -1 || v > bestVal {
			best, bestVal = i, v
		}
	}
	return best
}

// ApplyTopP ports apply_top_p (nucleus filtering): keep the smallest set of
// tokens whose cumulative probability mass reaches top_p, mask the rest to -inf.
// It reproduces the reference exactly — exponentiate, sort ascending, take the
// running cumulative mass, scatter it back to the original positions, and keep a
// token when its cumulative-from-the-bottom mass exceeds 1 - top_p.
func ApplyTopP(logits []float32, topP float64) {
	n := len(logits)
	if n == 0 {
		return
	}
	probs := make([]float64, n)
	for i, v := range logits {
		probs[i] = math.Exp(float64(v))
	}
	order := make([]int, n) // ascending by logit
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] < logits[order[b]] })

	// cumAtPos[i] = sum of probs of all tokens with logit <= logits[i], in the
	// ascending sweep including i itself.
	cumAtPos := make([]float64, n)
	var run float64
	for _, idx := range order {
		run += probs[idx]
		cumAtPos[idx] = run
	}
	threshold := 1.0 - topP
	for i := range logits {
		if !(cumAtPos[i] > threshold) {
			logits[i] = negInf
		}
	}
}

// ApplyMinP ports apply_min_p: drop tokens whose log-prob is below
// max + log(min_p), while always keeping the highest minTokensToKeep tokens. A
// minTokensToKeep <= 1 means only the implicit single max is guaranteed (the max
// is never below the threshold). Mirrors the reference's argpartition keep-set.
func ApplyMinP(logits []float32, minP float64, minTokensToKeep int) {
	n := len(logits)
	if n == 0 || !(minP > 0 && minP <= 1.0) {
		return
	}
	top := logits[Argmax(logits)]
	scaled := float64(top) + math.Log(minP)
	remove := make([]bool, n)
	for i, v := range logits {
		remove[i] = float64(v) < scaled
	}
	if minTokensToKeep > 1 {
		for _, idx := range topIndices(logits, minTokensToKeep) {
			remove[idx] = false
		}
	}
	for i := range logits {
		if remove[i] {
			logits[i] = negInf
		}
	}
}

// ApplyTopK ports apply_top_k: keep the top_k highest-probability tokens, mask
// the rest to -inf.
func ApplyTopK(logits []float32, topK int) {
	n := len(logits)
	if n == 0 || topK <= 0 || topK >= n {
		return
	}
	keep := make([]bool, n)
	for _, idx := range topIndices(logits, topK) {
		keep[idx] = true
	}
	for i := range logits {
		if !keep[i] {
			logits[i] = negInf
		}
	}
}

// topIndices returns the indices of the k largest logits, ties broken toward the
// lower index for determinism.
func topIndices(logits []float32, k int) []int {
	n := len(logits)
	if k > n {
		k = n
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		if logits[order[a]] != logits[order[b]] {
			return logits[order[a]] > logits[order[b]]
		}
		return order[a] < order[b]
	})
	return order[:k]
}
