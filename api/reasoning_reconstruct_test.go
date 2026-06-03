// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type reconstructFixture struct {
	Reconstruct []struct {
		Role         string          `json:"role"`
		Content      json.RawMessage `json:"content"`
		Reasoning    *string         `json:"reasoning"`
		Native       bool            `json:"native"`
		NewContent   json.RawMessage `json:"new_content"`
		ReasoningOut *string         `json:"reasoning_out"`
	} `json:"reconstruct"`
}

func loadReconstruct(t *testing.T) reconstructFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/reasoning_reconstruct.json")
	if err != nil {
		t.Fatal(err)
	}
	var f reconstructFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestApplyReasoningReconstructionParity(t *testing.T) {
	for i, c := range loadReconstruct(t).Reconstruct {
		content, ok := parseOrdered(string(c.Content))
		if !ok {
			t.Fatalf("case %d: bad content fixture", i)
		}
		reasoning := ""
		if c.Reasoning != nil {
			reasoning = *c.Reasoning
		}
		newContent, reasoningOut, hasReasoning := applyReasoningReconstruction(c.Role, content, reasoning, c.Native)

		want, ok := parseOrdered(string(c.NewContent))
		if !ok {
			t.Fatalf("case %d: bad new_content fixture", i)
		}
		if newContent.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d new_content:\n got  %s\n want %s", i, newContent.dumpASCII(), want.dumpASCII())
		}

		// reasoning_out is null in the fixture exactly when no reasoning attaches.
		if c.ReasoningOut == nil {
			if hasReasoning {
				t.Errorf("case %d: got reasoning %q, want none", i, reasoningOut)
			}
		} else {
			if !hasReasoning {
				t.Errorf("case %d: got no reasoning, want %q", i, *c.ReasoningOut)
			} else if reasoningOut != *c.ReasoningOut {
				t.Errorf("case %d reasoning_out:\n got  %q\n want %q", i, reasoningOut, *c.ReasoningOut)
			}
		}
	}
}

func BenchmarkApplyReasoningReconstruction(b *testing.B) {
	content, _ := parseOrdered(`[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]`)
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = applyReasoningReconstruction("assistant", content, "thinking hard", false)
	}
}
