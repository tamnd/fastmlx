// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"encoding/json"
	"os"
	"testing"
)

type vlmKeysFixture struct {
	Keys []struct {
		VLMImageHash string   `json:"vlm_image_hash"`
		Out          []string `json:"out"`
	} `json:"keys"`
	Start []struct {
		VLMImageHash     string `json:"vlm_image_hash"`
		VLMCacheKeyStart int    `json:"vlm_cache_key_start"`
		Out              *int   `json:"out"`
	} `json:"start"`
	Ranges []struct {
		VLMCacheKeyRanges [][]json.RawMessage `json:"vlm_cache_key_ranges"`
		Out               [][]json.RawMessage `json:"out"`
	} `json:"ranges"`
}

func loadVLMKeys(t *testing.T) vlmKeysFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/vlm_keys.json")
	if err != nil {
		t.Fatal(err)
	}
	var f vlmKeysFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestVLMExtraKeysForCacheParity(t *testing.T) {
	for i, c := range loadVLMKeys(t).Keys {
		got := VLMExtraKeysForCache(c.VLMImageHash)
		if !sameStrings(got, c.Out) {
			t.Errorf("keys case %d: got %v, want %v", i, got, c.Out)
		}
	}
}

func TestVLMExtraKeyTokenStartForCacheParity(t *testing.T) {
	for i, c := range loadVLMKeys(t).Start {
		gotStart, gotOK := VLMExtraKeyTokenStartForCache(c.VLMImageHash, c.VLMCacheKeyStart)
		if c.Out == nil {
			if gotOK {
				t.Errorf("start case %d: got %d, want none", i, gotStart)
			}
		} else if !gotOK || gotStart != *c.Out {
			t.Errorf("start case %d: got (%d,%v), want %d", i, gotStart, gotOK, *c.Out)
		}
	}
}

func TestVLMExtraKeyRangesForCacheParity(t *testing.T) {
	for i, c := range loadVLMKeys(t).Ranges {
		ranges := make([]VLMCacheKeyRange, len(c.VLMCacheKeyRanges))
		for j, pair := range c.VLMCacheKeyRanges {
			var start int
			var hash string
			if err := json.Unmarshal(pair[0], &start); err != nil {
				t.Fatalf("ranges case %d: bad start: %v", i, err)
			}
			if err := json.Unmarshal(pair[1], &hash); err != nil {
				t.Fatalf("ranges case %d: bad hash: %v", i, err)
			}
			ranges[j] = VLMCacheKeyRange{TokenStart: start, ImageHash: hash}
		}
		got := VLMExtraKeyRangesForCache(ranges)
		if len(got) != len(c.Out) {
			t.Errorf("ranges case %d: got %d segments, want %d", i, len(got), len(c.Out))
			continue
		}
		for j, seg := range got {
			var wantStart int
			var wantKeys []string
			_ = json.Unmarshal(c.Out[j][0], &wantStart)
			_ = json.Unmarshal(c.Out[j][1], &wantKeys)
			if seg.TokenStart != wantStart || !sameStrings(seg.ExtraKeys, wantKeys) {
				t.Errorf("ranges case %d seg %d: got (%d,%v), want (%d,%v)",
					i, j, seg.TokenStart, seg.ExtraKeys, wantStart, wantKeys)
			}
		}
	}
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkVLMExtraKeyRangesForCache(b *testing.B) {
	ranges := []VLMCacheKeyRange{{0, "h0"}, {256, "h1"}, {512, "h2"}}
	b.ReportAllocs()
	for b.Loop() {
		_ = VLMExtraKeyRangesForCache(ranges)
	}
}
