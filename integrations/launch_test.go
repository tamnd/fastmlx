// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import (
	"encoding/json"
	"maps"
	"os"
	"testing"
)

type launchFixture struct {
	Commands []struct {
		Prefix string `json:"prefix"`
		Tool   string `json:"tool"`
		Model  string `json:"model"`
		Out    string `json:"out"`
	} `json:"commands"`
	Scrub []struct {
		In  map[string]string `json:"in"`
		Out map[string]string `json:"out"`
	} `json:"scrub"`
	Format []struct {
		Index            int    `json:"index"`
		ID               string `json:"id"`
		MaxContextWindow *int   `json:"max_context_window"`
		Out              string `json:"out"`
	} `json:"format"`
	Auto []struct {
		N           int    `json:"n"`
		ID          string `json:"id"`
		NeedsPrompt bool   `json:"needs_prompt"`
	} `json:"auto"`
	Choices []struct {
		Input string `json:"input"`
		N     int    `json:"n"`
		Out   []any  `json:"out"`
	} `json:"choices"`
}

func loadLaunch(t *testing.T) launchFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/launch.json")
	if err != nil {
		t.Fatal(err)
	}
	var f launchFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestLaunchCommandParity(t *testing.T) {
	f := loadLaunch(t)
	for i, c := range f.Commands {
		got := LaunchCommand(c.Prefix, c.Tool, c.Model)
		if got != c.Out {
			t.Errorf("LaunchCommand case %d = %q, want %q", i, got, c.Out)
		}
	}
}

func TestScrubbedEnvParity(t *testing.T) {
	f := loadLaunch(t)
	for i, c := range f.Scrub {
		got := ScrubbedEnv(c.In)
		if !maps.Equal(got, c.Out) {
			t.Errorf("ScrubbedEnv case %d = %v, want %v", i, got, c.Out)
		}
	}
}

func TestFormatModelOptionParity(t *testing.T) {
	f := loadLaunch(t)
	for i, c := range f.Format {
		got := FormatModelOption(c.Index, ModelInfo{ID: c.ID, MaxContextWindow: c.MaxContextWindow})
		if got != c.Out {
			t.Errorf("FormatModelOption case %d = %q, want %q", i, got, c.Out)
		}
	}
}

func TestAutoSelectModelParity(t *testing.T) {
	f := loadLaunch(t)
	for i, c := range f.Auto {
		models := make([]ModelInfo, c.N)
		for j := range models {
			models[j] = ModelInfo{ID: c.ID}
		}
		// A single-model set must carry the id the fixture recorded.
		if c.N == 1 {
			models[0] = ModelInfo{ID: c.ID}
		}
		id, needs := AutoSelectModel(models)
		if id != c.ID || needs != c.NeedsPrompt {
			t.Errorf("AutoSelectModel case %d = (%q,%v), want (%q,%v)", i, id, needs, c.ID, c.NeedsPrompt)
		}
	}
}

func TestParseModelChoiceParity(t *testing.T) {
	f := loadLaunch(t)
	for i, c := range f.Choices {
		idx, ok := ParseModelChoice(c.Input, c.N)
		wantIdx := int(c.Out[0].(float64))
		wantOK := c.Out[1].(bool)
		// The reference reports -1 for a rejected choice; the Go port reports a
		// zero index with ok=false, so only compare the index when ok is true.
		if ok != wantOK || (ok && idx != wantIdx) {
			t.Errorf("ParseModelChoice case %d (%q) = (%d,%v), want (%d,%v)", i, c.Input, idx, ok, wantIdx, wantOK)
		}
	}
}

func BenchmarkLaunchCommand(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = LaunchCommand("fastmlx", "codex", "Qwen3-8B")
	}
}

func BenchmarkFormatModelOption(b *testing.B) {
	cw := 262144
	m := ModelInfo{ID: "Qwen3-8B", MaxContextWindow: &cw}
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatModelOption(3, m)
	}
}
