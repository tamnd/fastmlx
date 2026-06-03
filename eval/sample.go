// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"math"
	"sort"
)

// SampleSeed is the fixed seed every dataset sampler uses, so the same questions
// are always selected and different models are compared on an identical subset.
const SampleSeed = 42

// Sample returns k items chosen from population in selection order using a
// freshly seeded Python-compatible generator, reproducing random.Random(seed).
// sample(population, k). The population is not modified.
func Sample[T any](population []T, k int, seed uint64) []T {
	r := NewPyRandom(seed)
	idx := r.SampleIndices(len(population), k)
	out := make([]T, k)
	for i, j := range idx {
		out[i] = population[j]
	}
	return out
}

// DeterministicSample returns n items sampled with the fixed seed, or the whole
// slice when n is at least the population size, matching the reference
// deterministic_sample.
func DeterministicSample[T any](items []T, n int) []T {
	if n >= len(items) {
		return items
	}
	return Sample(items, n, SampleSeed)
}

// StratifiedSample draws a size-n sample with proportional representation from
// each category, the category of an item given by key. It reproduces the
// reference stratified_sample exactly: one shared fixed-seed generator across
// all groups, categories visited in sorted order, each non-final category
// allocated max(1, round(len(group)/total*n)) capped by the remaining budget and
// the group size, and the final category taking whatever budget remains. Returns
// the whole slice when n is at least the population size.
func StratifiedSample[T any](items []T, n int, key func(T) string) []T {
	if n >= len(items) {
		return items
	}
	r := NewPyRandom(SampleSeed)

	groups := map[string][]T{}
	var order []string
	for _, it := range items {
		cat := key(it)
		if _, ok := groups[cat]; !ok {
			order = append(order, cat)
		}
		groups[cat] = append(groups[cat], it)
	}
	sort.Strings(order)

	total := len(items)
	var sampled []T
	remaining := n
	for i, cat := range order {
		group := groups[cat]
		var count int
		if i == len(order)-1 {
			count = remaining
		} else {
			count = int(math.Max(1, roundHalfEven(float64(len(group))/float64(total)*float64(n))))
			count = min(count, remaining, len(group))
		}
		take := min(count, len(group))
		idx := r.SampleIndices(len(group), take)
		for _, j := range idx {
			sampled = append(sampled, group[j])
		}
		remaining -= take
		if remaining <= 0 {
			break
		}
	}
	return sampled
}

// roundHalfEven reproduces Python's round() for a single float argument, which
// rounds half to even.
func roundHalfEven(x float64) float64 {
	return math.RoundToEven(x)
}
