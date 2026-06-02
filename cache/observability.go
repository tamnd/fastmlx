// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"fmt"
	"maps"
	"sync"
)

// RateTracker keeps a ring of counter snapshots and derives per-window rates
// from them, so the dashboard can show "prefix hits over the last minute"
// without the cache cores tracking time. Counters are opaque integer maps; the
// tracker computes deltas between a window's baseline snapshot and the newest.
//
// The wall clock is injected (seconds, monotonic) so the windowing is
// deterministic under test; the reference uses time.monotonic().
type RateTracker struct {
	mu           sync.Mutex
	snapshots    []rateSnapshot
	maxSnapshots int
	minInterval  float64
	now          func() float64
}

type rateSnapshot struct {
	ts       float64
	counters map[string]int
}

// Default windowing, matching the reference.
const (
	DefaultMaxSnapshots = 90
	DefaultMinInterval  = 10.0
)

// DefaultWindows are the rate windows in seconds (1m, 5m, 15m).
var DefaultWindows = []int{60, 300, 900}

// NewRateTracker builds a tracker. A non-positive maxSnapshots or now func uses
// the defaults; the clock must be supplied for deterministic behavior.
func NewRateTracker(maxSnapshots int, minInterval float64, now func() float64) *RateTracker {
	if maxSnapshots <= 0 {
		maxSnapshots = DefaultMaxSnapshots
	}
	return &RateTracker{
		maxSnapshots: maxSnapshots,
		minInterval:  minInterval,
		now:          now,
	}
}

// MaybeSnapshot records the counters unless the most recent snapshot is younger
// than the minimum interval. It returns whether a snapshot was taken. The
// counters are copied, so later mutation of the caller's map is not observed.
func (t *RateTracker) MaybeSnapshot(counters map[string]int) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := t.now()
	if len(t.snapshots) > 0 && (now-t.snapshots[len(t.snapshots)-1].ts) < t.minInterval {
		return false
	}
	cp := make(map[string]int, len(counters))
	maps.Copy(cp, counters)
	t.snapshots = append(t.snapshots, rateSnapshot{ts: now, counters: cp})
	if len(t.snapshots) > t.maxSnapshots {
		t.snapshots = t.snapshots[len(t.snapshots)-t.maxSnapshots:]
	}
	return true
}

// Clear drops all snapshots.
func (t *RateTracker) Clear() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.snapshots = nil
}

// GetRates returns the JSON object {"windows": {...}, "cumulative": {...}} for
// the requested windows. With no snapshots both are empty. A window whose
// baseline is less than a second old reports an empty object, matching the
// reference's elapsed < 1.0 guard.
func (t *RateTracker) GetRates(windows []int) string {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.snapshots) == 0 {
		return `{"windows":{},"cumulative":{}}`
	}

	now := t.snapshots[len(t.snapshots)-1].ts
	newest := t.snapshots[len(t.snapshots)-1].counters

	var windowPairs []kv
	for _, w := range windows {
		label := windowLabel(w)
		var baselineTS float64
		var baselineCounters map[string]int
		found := false
		for _, s := range t.snapshots {
			if (now - s.ts) <= float64(w) {
				baselineTS, baselineCounters = s.ts, s.counters
				found = true
				break
			}
		}
		if !found {
			baselineTS, baselineCounters = t.snapshots[0].ts, t.snapshots[0].counters
		}
		elapsed := now - baselineTS
		if elapsed < 1.0 {
			windowPairs = append(windowPairs, kv{label, []kv{}})
			continue
		}
		windowPairs = append(windowPairs, kv{label, computeWindow(baselineCounters, newest, elapsed)})
	}

	return encodeOrdered([]kv{
		{"windows", windowPairs},
		{"cumulative", computeCumulative(newest)},
	})
}

// SnapshotAndGetRates records the counters then returns the current rates.
func (t *RateTracker) SnapshotAndGetRates(counters map[string]int, windows []int) string {
	t.MaybeSnapshot(counters)
	return t.GetRates(windows)
}

func windowLabel(seconds int) string {
	if seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	}
	return fmt.Sprintf("%dm", seconds/60)
}

func safeRatio(numerator, denominator int) float64 {
	if denominator == 0 {
		return 0.0
	}
	return float64(numerator) / float64(denominator)
}

func computeWindow(old, new map[string]int, elapsed float64) []kv {
	delta := func(key string) int {
		d := new[key] - old[key]
		if d < 0 {
			return 0
		}
		return d
	}

	dPrefixHits := delta("prefix_hits")
	dPrefixMisses := delta("prefix_misses")
	dEvictions := delta("evictions")
	dSSDHot := delta("ssd_hot_hits")
	dSSDDisk := delta("ssd_disk_loads")
	dTokensMatched := delta("prefix_tokens_matched")
	dTokensRequested := delta("prefix_tokens_requested")

	minutes := elapsed / 60.0
	evictionRate := 0.0
	if minutes > 0 {
		evictionRate = round2(float64(dEvictions) / minutes)
	}

	return []kv{
		{"prefix_hit_rate", round4(safeRatio(dPrefixHits, dPrefixHits+dPrefixMisses))},
		{"prefix_hits", dPrefixHits},
		{"prefix_misses", dPrefixMisses},
		{"prefix_match_efficiency", round4(safeRatio(dTokensMatched, dTokensRequested))},
		{"evictions", dEvictions},
		{"eviction_rate_per_min", evictionRate},
		{"ssd_hot_hits", dSSDHot},
		{"ssd_disk_loads", dSSDDisk},
		{"ssd_hot_rate", round4(safeRatio(dSSDHot, dSSDHot+dSSDDisk))},
	}
}

func computeCumulative(counters map[string]int) []kv {
	prefixHits := counters["prefix_hits"]
	prefixMisses := counters["prefix_misses"]
	ssdHot := counters["ssd_hot_hits"]
	ssdDisk := counters["ssd_disk_loads"]
	tokensMatched := counters["prefix_tokens_matched"]
	tokensRequested := counters["prefix_tokens_requested"]

	return []kv{
		{"prefix_hits", prefixHits},
		{"prefix_misses", prefixMisses},
		{"prefix_hit_rate", round4(safeRatio(prefixHits, prefixHits+prefixMisses))},
		{"prefix_tokens_saved", counters["prefix_tokens_saved"]},
		{"prefix_match_efficiency", round4(safeRatio(tokensMatched, tokensRequested))},
		{"evictions", counters["evictions"]},
		{"ssd_hot_hits", ssdHot},
		{"ssd_disk_loads", ssdDisk},
		{"ssd_saves", counters["ssd_saves"]},
		{"hot_cache_evictions", counters["hot_cache_evictions"]},
		{"hot_cache_promotions", counters["hot_cache_promotions"]},
		{"ssd_hot_rate", round4(safeRatio(ssdHot, ssdHot+ssdDisk))},
	}
}
