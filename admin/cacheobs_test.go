// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type cacheObsFixture struct {
	Entries  []map[string]any `json:"entries"`
	Payloads []map[string]any `json:"payloads"`
	Empty    map[string]any   `json:"empty"`
}

func loadCacheObs(t *testing.T) cacheObsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/cacheobs.json")
	if err != nil {
		t.Fatal(err)
	}
	var f cacheObsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// rsA..rsE mirror the native-typed entry cases in the capture script. Each call
// returns a fresh map so callers can mutate (e.g. inject "id") without aliasing.
func rsA() map[string]any {
	return map[string]any{
		"block_size":     16,
		"indexed_blocks": 100,
		"prefix_cache":   map[string]any{"block_size": 16, "partial_block_skips": 0},
		"ssd_cache": map[string]any{
			"num_files": 5, "total_size_bytes": 1000, "max_size_bytes": 4000,
			"hot_cache_max_bytes": 200, "hot_cache_size_bytes": 50, "hot_cache_entries": 3,
		},
		"cache_rates": map[string]any{"hit_rate": 0.5},
	}
}

func rsB() map[string]any {
	return map[string]any{
		"block_size":     0,
		"indexed_blocks": 0,
		"prefix_cache": map[string]any{
			"block_size": 32, "partial_block_skips": 3, "partial_tokens_skipped": 7,
			"last_partial_tokens_skipped": 2, "last_tokens_to_next_block": 9,
		},
		"ssd_cache": map[string]any{"num_files": 2, "total_size_bytes": 500, "max_size_bytes": 1000},
	}
}

func cacheEntryCases() []struct {
	id any
	rs map[string]any
} {
	return []struct {
		id any
		rs map[string]any
	}{
		{"model-a", rsA()},
		{"model-b", rsB()},
		{"model-c", map[string]any{
			"block_size": nil, "indexed_blocks": nil,
			"prefix_cache": map[string]any{}, "ssd_cache": map[string]any{},
		}},
		{"model-d", map[string]any{
			"block_size": 16.0, "indexed_blocks": 50,
			"prefix_cache": map[string]any{"block_size": 64}, "ssd_cache": map[string]any{"num_files": 1},
		}},
		{"model-e", map[string]any{
			"block_size": 8, "indexed_blocks": 10,
			"prefix_cache": nil, "ssd_cache": nil, "cache_rates": map[string]any{},
		}},
		{"model-f", map[string]any{
			"block_size": 16, "indexed_blocks": 4,
			"prefix_cache": map[string]any{"partial_tokens_skipped": 3.9},
			"ssd_cache":    map[string]any{"num_files": 2.9, "total_size_bytes": 999.7},
		}},
	}
}

func TestBuildCacheModelEntryParity(t *testing.T) {
	want := loadCacheObs(t).Entries
	cases := cacheEntryCases()
	if len(cases) != len(want) {
		t.Fatalf("case count %d != fixture count %d", len(cases), len(want))
	}
	for i, c := range cases {
		got := jsonRoundTrip(t, BuildCacheModelEntry(c.id, c.rs))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("BuildCacheModelEntry case %d (%v):\n got  %v\n want %v", i, c.id, got, want[i])
		}
	}
}

func TestBuildCacheObservabilityParity(t *testing.T) {
	want := loadCacheObs(t).Payloads

	withID := func(id string, rs map[string]any) map[string]any {
		rs["id"] = id
		return rs
	}
	cfg0 := CacheObsConfig{BasePath: "/base", SsdCacheDir: "/base/ssd",
		ResponseStateDir: "/base/ssd/response-state", DiskMaxBytes: 100000}

	cases := []struct {
		cfg    CacheObsConfig
		models []map[string]any
	}{
		{cfg0, nil},
		{cfg0, []map[string]any{withID("model-a", rsA()), withID("model-b", rsB())}},
		{CacheObsConfig{BasePath: "/x", SsdCacheDir: "/x/c",
			ResponseStateDir: "/x/c/response-state", DiskMaxBytes: 10},
			[]map[string]any{withID("model-a", rsA())}},
	}
	if len(cases) != len(want) {
		t.Fatalf("case count %d != fixture count %d", len(cases), len(want))
	}
	for i, c := range cases {
		got := jsonRoundTrip(t, BuildCacheObservability(c.cfg, c.models))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("BuildCacheObservability case %d:\n got  %v\n want %v", i, got, want[i])
		}
	}
}

func TestEmptyCacheObservabilityParity(t *testing.T) {
	got := jsonRoundTrip(t, EmptyCacheObservability())
	want := loadCacheObs(t).Empty
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EmptyCacheObservability:\n got  %v\n want %v", got, want)
	}
}

func BenchmarkBuildCacheModelEntry(b *testing.B) {
	rs := rsA()
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildCacheModelEntry("model-a", rs)
	}
}

func BenchmarkBuildCacheObservability(b *testing.B) {
	cfg := CacheObsConfig{BasePath: "/base", SsdCacheDir: "/base/ssd",
		ResponseStateDir: "/base/ssd/response-state", DiskMaxBytes: 100000}
	rs := rsA()
	rs["id"] = "model-a"
	models := []map[string]any{rs}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildCacheObservability(cfg, models)
	}
}
