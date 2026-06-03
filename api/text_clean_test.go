// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type textCleanFixture struct {
	CleanSpecial []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"clean_special"`
	CleanOutput []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"clean_output"`
	DetectPartial []struct {
		In        json.RawMessage `json:"in"`
		IsPartial bool            `json:"is_partial"`
		Out       json.RawMessage `json:"out"`
	} `json:"detect_partial"`
	Usage []struct {
		In  usageFields `json:"in"`
		Out usageFields `json:"out"`
	} `json:"usage"`
}

type usageFields struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
}

func loadTextCleanFixture(t *testing.T) textCleanFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/text_clean.json")
	if err != nil {
		t.Fatal(err)
	}
	var f textCleanFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCleanSpecialTokensParity(t *testing.T) {
	for _, c := range loadTextCleanFixture(t).CleanSpecial {
		if got := CleanSpecialTokens(c.In); got != c.Out {
			t.Errorf("CleanSpecialTokens(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestCleanOutputTextParity(t *testing.T) {
	for _, c := range loadTextCleanFixture(t).CleanOutput {
		if got := CleanOutputText(c.In); got != c.Out {
			t.Errorf("CleanOutputText(%q) = %q, want %q", c.In, got, c.Out)
		}
	}
}

func TestDetectAndStripPartialParity(t *testing.T) {
	for i, c := range loadTextCleanFixture(t).DetectPartial {
		in, ok := parseOrdered(string(c.In))
		if !ok {
			t.Fatalf("detect_partial[%d]: bad input", i)
		}
		gotFlag, gotMsgs := DetectAndStripPartial(in.arr)
		if gotFlag != c.IsPartial {
			t.Errorf("detect_partial[%d]: flag = %v, want %v", i, gotFlag, c.IsPartial)
		}
		want := parseFixture(t, c.Out).dump()
		got := jval{kind: kindArray, arr: gotMsgs}.dump()
		if got != want {
			t.Errorf("detect_partial[%d]: msgs = %s, want %s", i, got, want)
		}
	}
}

func TestNormalizeBaseUsageParity(t *testing.T) {
	for i, c := range loadTextCleanFixture(t).Usage {
		in := BaseUsage{
			PromptTokens:     c.In.PromptTokens,
			CompletionTokens: c.In.CompletionTokens,
			TotalTokens:      c.In.TotalTokens,
			InputTokens:      c.In.InputTokens,
			OutputTokens:     c.In.OutputTokens,
		}
		got := NormalizeBaseUsage(in)
		want := BaseUsage{
			PromptTokens:     c.Out.PromptTokens,
			CompletionTokens: c.Out.CompletionTokens,
			TotalTokens:      c.Out.TotalTokens,
			InputTokens:      c.Out.InputTokens,
			OutputTokens:     c.Out.OutputTokens,
		}
		if got != want {
			t.Errorf("usage[%d]: got %+v, want %+v", i, got, want)
		}
	}
}

func BenchmarkCleanOutputText(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = CleanOutputText("<|im_start|><think>hidden</think>visible<|im_end|>")
	}
}
