// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type directCase struct {
	DirectURL  map[string]any `json:"direct_url"`
	DefaultURL string         `json:"default_url"`
	Out        map[string]any `json:"out"`
}

type engineCase struct {
	Data     map[string]any    `json:"data"`
	Packages map[string]string `json:"packages"`
	Out      map[string]any    `json:"out"`
}

type pyprojectCase struct {
	Content  string            `json:"content"`
	Packages map[string]string `json:"packages"`
	Out      map[string]any    `json:"out"`
}

type commitsFixture struct {
	Direct    []directCase    `json:"direct"`
	Engine    []engineCase    `json:"engine"`
	Pyproject []pyprojectCase `json:"pyproject"`
}

func loadCommits(t *testing.T) commitsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/commits.json")
	if err != nil {
		t.Fatal(err)
	}
	var f commitsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func jsonRoundTripAny(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestCommitFromDirectURLParity(t *testing.T) {
	for i, c := range loadCommits(t).Direct {
		got, ok := CommitFromDirectURL(c.DirectURL, c.DefaultURL)
		if c.Out == nil {
			if ok {
				t.Errorf("CommitFromDirectURL case %d: got %v, want nil", i, got)
			}
			continue
		}
		if !ok {
			t.Errorf("CommitFromDirectURL case %d: got nil, want %v", i, c.Out)
			continue
		}
		if rt := jsonRoundTrip(t, got); !reflect.DeepEqual(rt, c.Out) {
			t.Errorf("CommitFromDirectURL case %d:\n got  %v\n want %v", i, rt, c.Out)
		}
	}
}

func TestCommitsFromEngineDataParity(t *testing.T) {
	for i, c := range loadCommits(t).Engine {
		got := jsonRoundTripAny(t, CommitsFromEngineData(c.Data, c.Packages))
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("CommitsFromEngineData case %d:\n got  %v\n want %v", i, got, c.Out)
		}
	}
}

func TestParseCommitsFromPyprojectParity(t *testing.T) {
	for i, c := range loadCommits(t).Pyproject {
		got := jsonRoundTripAny(t, ParseCommitsFromPyproject(c.Content, c.Packages))
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("ParseCommitsFromPyproject case %d:\n got  %v\n want %v", i, got, c.Out)
		}
	}
}

func BenchmarkParseCommitsFromPyproject(b *testing.B) {
	content := `dependencies = [
    "mlx-lm @ git+https://github.com/ml-explore/mlx-lm@abc1234567",
    "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@deadbeef0011",
]`
	packages := map[string]string{
		"mlx-lm":  "https://github.com/ml-explore/mlx-lm",
		"mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ParseCommitsFromPyproject(content, packages)
	}
}
