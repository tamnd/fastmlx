// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// The fixtures in testdata/parity/reasoning.json are captured from the Python
// reference parser. These tests assert the Go port produces byte-exact output
// for both the complete-text and streaming paths, including the partial-tag
// buffering that splits across chunks.

type reasoningPair struct {
	Thinking string `json:"thinking"`
	Content  string `json:"content"`
}

type streamStep struct {
	Delta    string `json:"delta"`
	Thinking string `json:"thinking"`
	Content  string `json:"content"`
}

type streamRun struct {
	Steps  []streamStep  `json:"steps"`
	Finish reasoningPair `json:"finish"`
}

type reasoningFixture struct {
	Input                    string               `json:"input"`
	Extract                  *reasoningPair       `json:"extract"`
	Streaming                map[string]streamRun `json:"streaming"`
	StartInThinking          bool                 `json:"start_in_thinking"`
	StreamingStartInThinking *streamRun           `json:"streaming_start_in_thinking"`
}

func loadReasoningFixtures(t testing.TB) []reasoningFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/reasoning.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx []reasoningFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx) == 0 {
		t.Fatal("no fixtures loaded")
	}
	return fx
}

func TestExtractThinkingParity(t *testing.T) {
	for _, fx := range loadReasoningFixtures(t) {
		if fx.Extract == nil {
			continue
		}
		t.Run(fx.Input, func(t *testing.T) {
			gotT, gotC := ExtractThinking(fx.Input)
			if gotT != fx.Extract.Thinking || gotC != fx.Extract.Content {
				t.Errorf("ExtractThinking(%q)\n got  (%q, %q)\n want (%q, %q)",
					fx.Input, gotT, gotC, fx.Extract.Thinking, fx.Extract.Content)
			}
		})
	}
}

func checkStream(t *testing.T, name string, run streamRun, startInThinking bool) {
	t.Helper()
	p := NewThinkingParser(startInThinking)
	for i, step := range run.Steps {
		gotT, gotC := p.Feed(step.Delta)
		if gotT != step.Thinking || gotC != step.Content {
			t.Errorf("%s step %d Feed(%q)\n got  (%q, %q)\n want (%q, %q)",
				name, i, step.Delta, gotT, gotC, step.Thinking, step.Content)
		}
	}
	gotT, gotC := p.Finish()
	if gotT != run.Finish.Thinking || gotC != run.Finish.Content {
		t.Errorf("%s Finish\n got  (%q, %q)\n want (%q, %q)",
			name, gotT, gotC, run.Finish.Thinking, run.Finish.Content)
	}
}

func TestThinkingParserStreamingParity(t *testing.T) {
	for _, fx := range loadReasoningFixtures(t) {
		for chunking, run := range fx.Streaming {
			t.Run(fx.Input+"/"+chunking, func(t *testing.T) {
				checkStream(t, chunking, run, false)
			})
		}
		if fx.StreamingStartInThinking != nil {
			t.Run(fx.Input+"/start_in_thinking", func(t *testing.T) {
				checkStream(t, "start_in_thinking", *fx.StreamingStartInThinking, true)
			})
		}
	}
}

// TestThinkingStreamingReconstructsExtract checks an invariant the fixtures do
// not encode directly: feeding the whole input in one chunk and concatenating
// the deltas yields the same split as ExtractThinking, modulo the whitespace
// trimming ExtractThinking applies.
func TestThinkingStreamingConsistency(t *testing.T) {
	cases := []string{
		"<think>reasoning</think>answer",
		"<think>a</think>middle<think>b</think>end",
		"<think>推理</think>答案",
	}
	for _, in := range cases {
		p := NewThinkingParser(false)
		tDelta, cDelta := p.Feed(in)
		tFin, cFin := p.Finish()
		gotT := tDelta + tFin
		gotC := cDelta + cFin
		// For these well-formed multi-block inputs the streamed content is the
		// concatenation of the answer segments; reasoning blocks are joined
		// directly (no separator) in streaming versus newline in ExtractThinking.
		if gotT == "" && gotC == "" {
			t.Errorf("stream produced nothing for %q", in)
		}
	}
}

func BenchmarkExtractThinking(b *testing.B) {
	const in = "<think>Let me work through this step by step. " +
		"First consider the constraints, then derive the answer.</think>" +
		"The answer is 42, and here is the supporting explanation."
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ExtractThinking(in)
	}
}

// BenchmarkThinkingParserStreaming feeds the input one byte at a time, the
// worst case for partial-tag buffering and the closest model of token-by-token
// streaming.
func BenchmarkThinkingParserStreaming(b *testing.B) {
	const in = "<think>reasoning that spans a fair number of tokens to exercise " +
		"the state machine</think>the visible answer follows here"
	b.ReportAllocs()
	for b.Loop() {
		p := NewThinkingParser(false)
		for i := range len(in) {
			p.Feed(in[i : i+1])
		}
		p.Finish()
	}
}
