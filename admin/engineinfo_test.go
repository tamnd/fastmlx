// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type engineInfoPackage struct {
	Name         string         `json:"name"`
	DistFound    bool           `json:"dist_found"`
	Version      string         `json:"version"`
	DirectCommit map[string]any `json:"direct_commit"`
}

type engineInfoCase struct {
	Packages []engineInfoPackage       `json:"packages"`
	Fallback map[string]map[string]any `json:"fallback"`
	Out      map[string]map[string]any `json:"out"`
}

type engineInfoFixture struct {
	Cases []engineInfoCase `json:"cases"`
}

func loadEngineInfo(t *testing.T) engineInfoFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/engineinfo.json")
	if err != nil {
		t.Fatal(err)
	}
	var f engineInfoFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBuildEngineInfoParity(t *testing.T) {
	for i, c := range loadEngineInfo(t).Cases {
		pkgs := make([]EnginePackageInfo, len(c.Packages))
		for j, p := range c.Packages {
			pkgs[j] = EnginePackageInfo{
				Name:         p.Name,
				DistFound:    p.DistFound,
				Version:      p.Version,
				DirectCommit: p.DirectCommit,
			}
		}
		got := jsonRoundTripAny(t, BuildEngineInfo(pkgs, c.Fallback))
		want := jsonRoundTripAny(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("BuildEngineInfo case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func BenchmarkBuildEngineInfo(b *testing.B) {
	pkgs := []EnginePackageInfo{
		{Name: "mlx-lm", DistFound: true, Version: "0.20.1",
			DirectCommit: map[string]any{"commit": "abc1234", "url": "https://x/commit/abc1234"}},
		{Name: "mlx-vlm", DistFound: true, Version: "0.1.5"},
		{Name: "mlx-audio", DistFound: false},
	}
	fallback := map[string]map[string]any{
		"mlx-vlm": {"commit": "def5678", "url": "https://y/commit/def5678"},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildEngineInfo(pkgs, fallback)
	}
}
