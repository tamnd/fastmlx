// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import (
	"encoding/json"
	"os"
	"testing"
)

type codexHermesFixture struct {
	Codex []struct {
		Ctx      rawCase `json:"ctx"`
		Existing string  `json:"existing"`
		Out      string  `json:"out"`
	} `json:"codex"`
	Hermes []struct {
		Ctx      rawCase        `json:"ctx"`
		Existing map[string]any `json:"existing"`
		Out      map[string]any `json:"out"`
	} `json:"hermes"`
}

func loadCodexHermes(t *testing.T) codexHermesFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/codexhermes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f codexHermesFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCodexConfigParity(t *testing.T) {
	f := loadCodexHermes(t)
	for i, c := range f.Codex {
		got := CodexConfig(c.Existing, c.Ctx.context())
		if got != c.Out {
			t.Errorf("CodexConfig case %d =\n%q\nwant\n%q", i, got, c.Out)
		}
	}
}

func TestHermesConfigParity(t *testing.T) {
	f := loadCodexHermes(t)
	for i, c := range f.Hermes {
		got := HermesConfig(cloneMap(t, c.Existing), c.Ctx.context())
		checkConfig(t, "HermesConfig", i, got, c.Out)
	}
}

func BenchmarkCodexConfig(b *testing.B) {
	c := Context{Host: "127.0.0.1", Port: 8000, APIKey: "k", Model: "Qwen3-8B"}
	existing := "model = \"old\"\napproval_policy = \"on-request\"\n\n[foo]\nbar = 1\n"
	b.ReportAllocs()
	for b.Loop() {
		_ = CodexConfig(existing, c)
	}
}

func BenchmarkHermesConfig(b *testing.B) {
	cw, mt := 128000, 4096
	c := Context{Host: "127.0.0.1", Port: 8000, APIKey: "k", Model: "Qwen3-8B", ContextWindow: &cw, MaxTokens: &mt}
	b.ReportAllocs()
	for b.Loop() {
		_ = HermesConfig(map[string]any{}, c)
	}
}
