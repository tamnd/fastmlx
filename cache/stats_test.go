// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type cachestatsFixture struct {
	Stats map[string]json.RawMessage `json:"stats"`
	Rates map[string]json.RawMessage `json:"rates"`
}

func loadCachestatsFixture(t *testing.T) cachestatsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/cachestats.json")
	if err != nil {
		t.Fatal(err)
	}
	var f cachestatsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// sameJSON parses two JSON documents and compares them structurally, so key
// order and float formatting differences do not matter (both decode to the same
// value tree).
func sameJSON(t *testing.T, got, want string) {
	t.Helper()
	var g, w any
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Fatalf("got is not valid JSON: %v\n%s", err, got)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want is not valid JSON: %v\n%s", err, want)
	}
	if !reflect.DeepEqual(g, w) {
		t.Errorf("JSON mismatch:\n got  %s\n want %s", got, want)
	}
}

func TestStatsSnapshotParity(t *testing.T) {
	fx := loadCachestatsFixture(t)

	prefix := &PrefixStats{
		BaseStats:                BaseStats{Hits: 7, Misses: 3, Evictions: 2},
		TokensSaved:              128,
		PartialBlockSkips:        4,
		PartialTokensSkipped:     40,
		BlockSize:                64,
		LastPartialTokensSkipped: 10,
		LastTokensToNextBlock:    54,
		TokensMatchedTotal:       200,
		TokensRequestedTotal:     260,
	}
	sameJSON(t, prefix.Snapshot(), string(fx.Stats["prefix"]))

	prefixSet := &PrefixStats{BaseStats: BaseStats{Hits: 1, Misses: 1}}
	prefixSet.SetTotalQueries(50)
	sameJSON(t, prefixSet.Snapshot(), string(fx.Stats["prefix_set_total"]))

	paged := &PagedStats{
		BaseStats:         BaseStats{Hits: 10, Misses: 0, Evictions: 1},
		TotalBlocks:       256,
		AllocatedBlocks:   64,
		FreeBlocks:        192,
		SharedBlocks:      8,
		TotalTokensCached: 4096,
		COWCopies:         3,
	}
	sameJSON(t, paged.Snapshot(), string(fx.Stats["paged"]))

	pagedEmpty := &PagedStats{}
	sameJSON(t, pagedEmpty.Snapshot(), string(fx.Stats["paged_empty"]))

	vlm := &VLMStats{
		BaseStats:      BaseStats{Hits: 5, Misses: 5},
		TokensSaved:    512,
		ImageCacheHits: 3,
	}
	sameJSON(t, vlm.Snapshot(), string(fx.Stats["vlm"]))

	ssd := &PagedSSDStats{
		BaseStats:              BaseStats{Hits: 20, Misses: 4, Evictions: 2},
		Saves:                  18,
		Loads:                  20,
		Errors:                 2,
		SSDWriteDrops:          1,
		TotalSizeBytes:         1048576,
		MaxSizeBytes:           2097152,
		ConfiguredMaxSizeBytes: 2097152,
		NumFiles:               12,
		HotCacheEntries:        6,
		HotCacheSizeBytes:      65536,
		HotCacheMaxBytes:       131072,
		HotCacheHits:           9,
		HotCacheEvictions:      3,
		HotCachePromotions:     2,
	}
	sameJSON(t, ssd.Snapshot(), string(fx.Stats["ssd"]))

	ssdEmpty := &PagedSSDStats{}
	sameJSON(t, ssdEmpty.Snapshot(), string(fx.Stats["ssd_empty"]))
}

func TestStatsComputedProperties(t *testing.T) {
	// total_queries prefers the explicit counter when set.
	p := &PrefixStats{BaseStats: BaseStats{Hits: 3, Misses: 1}}
	if p.TotalQueries() != 4 {
		t.Errorf("default total_queries = %d, want 4", p.TotalQueries())
	}
	p.SetTotalQueries(50)
	if p.TotalQueries() != 50 {
		t.Errorf("explicit total_queries = %d, want 50", p.TotalQueries())
	}

	// RecordLoad bumps both loads and hits.
	s := &PagedSSDStats{}
	s.RecordLoad()
	if s.Loads != 1 || s.Hits != 1 {
		t.Errorf("RecordLoad: loads=%d hits=%d, want 1/1", s.Loads, s.Hits)
	}

	// save_rate divides by saves+errors.
	s2 := &PagedSSDStats{Saves: 3, Errors: 1}
	if s2.SaveRate() != 0.75 {
		t.Errorf("save_rate = %v, want 0.75", s2.SaveRate())
	}
	if (&PagedSSDStats{}).SaveRate() != 0.0 {
		t.Error("empty save_rate should be 0")
	}

	// Reset keeps capacity, clears runtime.
	pg := &PagedStats{BaseStats: BaseStats{Hits: 5}, TotalBlocks: 256, COWCopies: 9}
	pg.Reset()
	if pg.Hits != 0 || pg.COWCopies != 0 || pg.TotalBlocks != 256 {
		t.Errorf("Reset: hits=%d cow=%d total=%d, want 0/0/256", pg.Hits, pg.COWCopies, pg.TotalBlocks)
	}
}

func TestRateTrackerParity(t *testing.T) {
	fx := loadCachestatsFixture(t)
	clock := &fakeClock{}
	now := clock.now

	// Empty tracker.
	empty := NewRateTracker(90, 10.0, now)
	sameJSON(t, empty.GetRates(DefaultWindows), string(fx.Rates["empty"]))

	// Two snapshots 120s apart.
	clock.t = 0.0
	tt := NewRateTracker(90, 10.0, now)
	tt.MaybeSnapshot(map[string]int{"prefix_hits": 0, "prefix_misses": 0, "evictions": 0,
		"ssd_hot_hits": 0, "ssd_disk_loads": 0, "prefix_tokens_matched": 0,
		"prefix_tokens_requested": 0, "prefix_tokens_saved": 0, "ssd_saves": 0,
		"hot_cache_evictions": 0, "hot_cache_promotions": 0})
	clock.t = 120.0
	if !tt.MaybeSnapshot(map[string]int{"prefix_hits": 80, "prefix_misses": 20, "evictions": 6,
		"ssd_hot_hits": 30, "ssd_disk_loads": 10, "prefix_tokens_matched": 700,
		"prefix_tokens_requested": 1000, "prefix_tokens_saved": 650, "ssd_saves": 40,
		"hot_cache_evictions": 5, "hot_cache_promotions": 7}) {
		t.Fatal("second snapshot 120s later should be recorded")
	}
	sameJSON(t, tt.GetRates(DefaultWindows), string(fx.Rates["two_120s"]))

	// min_interval suppression.
	clock.t = 0.0
	sup := NewRateTracker(90, 10.0, now)
	sup.MaybeSnapshot(map[string]int{"prefix_hits": 1, "prefix_misses": 1})
	clock.t = 5.0
	if sup.MaybeSnapshot(map[string]int{"prefix_hits": 99, "prefix_misses": 99}) {
		t.Error("snapshot within min_interval should be suppressed")
	}

	// Sub-second window reports an empty object for that window.
	clock.t = 0.0
	sub := NewRateTracker(90, 0.0, now)
	sub.MaybeSnapshot(map[string]int{"prefix_hits": 0, "prefix_misses": 0, "evictions": 0,
		"ssd_hot_hits": 0, "ssd_disk_loads": 0, "prefix_tokens_matched": 0,
		"prefix_tokens_requested": 0})
	clock.t = 0.5
	sub.MaybeSnapshot(map[string]int{"prefix_hits": 5, "prefix_misses": 1, "evictions": 0,
		"ssd_hot_hits": 0, "ssd_disk_loads": 0, "prefix_tokens_matched": 0,
		"prefix_tokens_requested": 0})
	sameJSON(t, sub.GetRates([]int{60}), string(fx.Rates["sub_second"]))
}

func TestRateTrackerRingEviction(t *testing.T) {
	clock := &fakeClock{}
	tr := NewRateTracker(3, 0.0, clock.now)
	for i := range 5 {
		clock.t = float64(i)
		tr.MaybeSnapshot(map[string]int{"prefix_hits": i})
	}
	// Only the last 3 snapshots survive (ts 2,3,4). The oldest baseline for a
	// wide window is therefore ts=2, not ts=0.
	if len(tr.snapshots) != 3 {
		t.Fatalf("ring kept %d snapshots, want 3", len(tr.snapshots))
	}
	if tr.snapshots[0].ts != 2.0 {
		t.Errorf("oldest surviving ts = %v, want 2", tr.snapshots[0].ts)
	}
}

func TestRateTrackerClear(t *testing.T) {
	clock := &fakeClock{}
	tr := NewRateTracker(90, 0.0, clock.now)
	tr.MaybeSnapshot(map[string]int{"prefix_hits": 1})
	tr.Clear()
	sameJSON(t, tr.GetRates(DefaultWindows), `{"windows":{},"cumulative":{}}`)
}

func TestRound(t *testing.T) {
	// Ties go to even, matching Python round.
	cases := []struct {
		in   float64
		n    int
		want float64
	}{
		{0.83333, 4, 0.8333},
		{0.66666, 4, 0.6667},
		{2.5, 0, 2.0},
		{3.5, 0, 4.0},
	}
	for _, c := range cases {
		if got := roundN(c.in, c.n); got != c.want {
			t.Errorf("roundN(%v, %d) = %v, want %v", c.in, c.n, got, c.want)
		}
	}
}

type fakeClock struct{ t float64 }

func (c *fakeClock) now() float64 { return c.t }

func BenchmarkRateTrackerGetRates(b *testing.B) {
	clock := &fakeClock{}
	tr := NewRateTracker(90, 0.0, clock.now)
	for i := range 90 {
		clock.t = float64(i * 11)
		tr.MaybeSnapshot(map[string]int{
			"prefix_hits": i * 10, "prefix_misses": i * 2, "evictions": i,
			"ssd_hot_hits": i * 3, "ssd_disk_loads": i, "prefix_tokens_matched": i * 100,
			"prefix_tokens_requested": i * 130,
		})
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = tr.GetRates(DefaultWindows)
	}
}
