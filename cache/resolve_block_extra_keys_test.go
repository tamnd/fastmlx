// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveBlockExtraKeysParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "resolve_block_extra_keys.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		BlockEnd           int      `json:"block_end"`
		ExtraKeys          []string `json:"extra_keys"`
		ExtraKeyTokenStart *int     `json:"extra_key_token_start"`
		ExtraKeyRanges     *[]struct {
			TokenStart int      `json:"token_start"`
			ExtraKeys  []string `json:"extra_keys"`
		} `json:"extra_key_ranges"`
		Result []string `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		var ranges []VLMCacheKeySegment
		if c.ExtraKeyRanges != nil {
			ranges = make([]VLMCacheKeySegment, len(*c.ExtraKeyRanges))
			for j, r := range *c.ExtraKeyRanges {
				ranges[j] = VLMCacheKeySegment{TokenStart: r.TokenStart, ExtraKeys: r.ExtraKeys}
			}
		}

		got := ResolveBlockExtraKeys(c.BlockEnd, c.ExtraKeys, c.ExtraKeyTokenStart, ranges)
		if !sameStrings(got, c.Result) {
			t.Errorf("case %d: ResolveBlockExtraKeys(%d, %v, %v, %v) got %v want %v",
				i, c.BlockEnd, c.ExtraKeys, c.ExtraKeyTokenStart, ranges, got, c.Result)
		}
	}
}

func BenchmarkResolveBlockExtraKeys(b *testing.B) {
	b.ReportAllocs()
	ranges := []VLMCacheKeySegment{
		{TokenStart: 0, ExtraKeys: []string{"imgA"}},
		{TokenStart: 20, ExtraKeys: []string{"imgB"}},
		{TokenStart: 40, ExtraKeys: []string{"imgC"}},
	}
	for b.Loop() {
		_ = ResolveBlockExtraKeys(30, nil, nil, ranges)
	}
}
