// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type docCase struct {
	Doc string   `json:"doc"`
	Out []string `json:"out"`
}

type sizeDetailedCase struct {
	Bytes int    `json:"bytes"`
	Out   string `json:"out"`
}

type grammarFixture struct {
	Docs  []docCase          `json:"docs"`
	Sizes []sizeDetailedCase `json:"sizes"`
}

func loadGrammar(t *testing.T) grammarFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/grammar.json")
	if err != nil {
		t.Fatal(err)
	}
	var f grammarFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestModelsFromDocstringParity(t *testing.T) {
	for i, c := range loadGrammar(t).Docs {
		got := ModelsFromDocstring(c.Doc)
		want := c.Out
		if want == nil {
			want = []string{}
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ModelsFromDocstring case %d (%q):\n got  %v\n want %v", i, c.Doc, got, want)
		}
	}
}

func TestFormatSizeDetailedParity(t *testing.T) {
	for i, c := range loadGrammar(t).Sizes {
		if got := FormatSizeDetailed(c.Bytes); got != c.Out {
			t.Errorf("FormatSizeDetailed case %d (%d) = %q, want %q", i, c.Bytes, got, c.Out)
		}
	}
}

func BenchmarkModelsFromDocstring(b *testing.B) {
	doc := "Supported models:\n- qwen3\n- llama3\n- gemma2\n- deepseek-v3\n"
	b.ReportAllocs()
	for b.Loop() {
		_ = ModelsFromDocstring(doc)
	}
}

func BenchmarkFormatSizeDetailed(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatSizeDetailed(8589934592)
	}
}
