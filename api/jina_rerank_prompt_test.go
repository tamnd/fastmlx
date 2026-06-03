// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type jinaRerankFixture struct {
	Sanitize []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"sanitize"`
	Format []struct {
		Query       string   `json:"query"`
		Documents   []string `json:"documents"`
		Instruction *string  `json:"instruction"`
		Out         string   `json:"out"`
	} `json:"format"`
}

func loadJinaRerankFixture(t *testing.T) jinaRerankFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/jina_rerank.json")
	if err != nil {
		t.Fatal(err)
	}
	var f jinaRerankFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSanitizeJinaTextParity(t *testing.T) {
	for i, c := range loadJinaRerankFixture(t).Sanitize {
		if got := sanitizeJinaText(c.In); got != c.Out {
			t.Errorf("case %d: sanitizeJinaText(%q) = %q, want %q", i, c.In, got, c.Out)
		}
	}
}

func TestFormatJinaRerankPromptParity(t *testing.T) {
	for i, c := range loadJinaRerankFixture(t).Format {
		instruction := ""
		if c.Instruction != nil {
			instruction = *c.Instruction
		}
		if got := FormatJinaRerankPrompt(c.Query, c.Documents, instruction); got != c.Out {
			t.Errorf("case %d:\n got  %q\n want %q", i, got, c.Out)
		}
	}
}

func BenchmarkFormatJinaRerankPrompt(b *testing.B) {
	docs := []string{"machine learning is a field of study", "the weather is sunny today", "pizza recipe"}
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatJinaRerankPrompt("what is machine learning?", docs, "prefer technical answers")
	}
}
