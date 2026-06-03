// SPDX-License-Identifier: MIT OR Apache-2.0

package release

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type updateResultFixture struct {
	Cases []struct {
		Selected *Release       `json:"selected"`
		Current  string         `json:"current"`
		Out      map[string]any `json:"out"`
	} `json:"cases"`
}

// updateRoundTrip decodes a value through JSON so the comparison sees the same
// types on both sides (numbers as float64, null as nil), sidestepping the Go
// vs Python numeric-repr differences.
func updateRoundTrip(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestBuildUpdateResultParity(t *testing.T) {
	data, err := os.ReadFile("testdata/updateresult.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx updateResultFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatal(err)
	}
	for i, c := range fx.Cases {
		got := updateRoundTrip(t, BuildUpdateResult(c.Selected, c.Current))
		want := updateRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("BuildUpdateResult case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func BenchmarkBuildUpdateResult(b *testing.B) {
	selected := &Release{TagName: "v0.5.0", HTMLURL: "https://x/releases/0.5.0"}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildUpdateResult(selected, "0.4.0")
	}
}
