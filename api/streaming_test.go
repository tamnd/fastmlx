// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestAppendEscapedMatchesEncodingJSON(t *testing.T) {
	cases := []string{
		"plain text",
		`quotes " and \ backslash`,
		"newline\ntab\tcr\r",
		"html <b> & </b>",
		"unicode héllo 日本語 🚀",
		"line sep para",
		"\x00\x01\x1f control",
	}
	for _, s := range cases {
		want, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		got := appendJSONString(nil, s)
		if !bytes.Equal(got, want) {
			t.Errorf("escape(%q):\n got  %s\n want %s", s, got, want)
		}
	}
}

func TestChunkEncoderContentDeltaParses(t *testing.T) {
	e := NewChunkEncoder("chatcmpl-123", "mock-model", 1700000000)
	var buf bytes.Buffer
	if err := e.WriteContentDelta(&buf, `hello "world" <tag>`); err != nil {
		t.Fatal(err)
	}
	line := buf.String()
	if !strings.HasPrefix(line, "data: ") || !strings.HasSuffix(line, "\n\n") {
		t.Fatalf("bad SSE framing: %q", line)
	}
	payload := strings.TrimSuffix(strings.TrimPrefix(line, "data: "), "\n\n")
	var chunk ChatCompletionChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatalf("payload not valid JSON: %v\n%s", err, payload)
	}
	if chunk.ID != "chatcmpl-123" || chunk.Model != "mock-model" || chunk.Created != 1700000000 {
		t.Errorf("header mismatch: %+v", chunk)
	}
	if len(chunk.Choices) != 1 || chunk.Choices[0].Delta.Content != `hello "world" <tag>` {
		t.Errorf("delta content mismatch: %+v", chunk.Choices)
	}
	if chunk.Choices[0].FinishReason != nil {
		t.Errorf("expected null finish_reason, got %v", *chunk.Choices[0].FinishReason)
	}
}

func TestChunkEncoderRoleAndFinish(t *testing.T) {
	e := NewChunkEncoder("id1", "m", 1)
	var buf bytes.Buffer
	if err := e.WriteRole(&buf); err != nil {
		t.Fatal(err)
	}
	usage := &Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8}
	if err := e.WriteFinish(&buf, "stop", usage); err != nil {
		t.Fatal(err)
	}
	if err := e.WriteDone(&buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, `"role":"assistant"`) {
		t.Error("missing role chunk")
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Error("missing finish_reason")
	}
	if !strings.Contains(out, `"total_tokens":8`) {
		t.Error("missing usage")
	}
	if !strings.HasSuffix(out, "data: [DONE]\n\n") {
		t.Error("missing DONE terminator")
	}
}

func TestStopFieldDecode(t *testing.T) {
	var single StopField
	if err := json.Unmarshal([]byte(`"x"`), &single); err != nil || len(single) != 1 || single[0] != "x" {
		t.Errorf("single decode: %v %v", single, err)
	}
	var arr StopField
	if err := json.Unmarshal([]byte(`["a","b"]`), &arr); err != nil || len(arr) != 2 {
		t.Errorf("array decode: %v %v", arr, err)
	}
	var null StopField
	if err := json.Unmarshal([]byte(`null`), &null); err != nil || null != nil {
		t.Errorf("null decode: %v %v", null, err)
	}
}

func BenchmarkWriteContentDelta(b *testing.B) {
	e := NewChunkEncoder("chatcmpl-bench", "mock-model", 1700000000)
	w := io.Discard
	b.ReportAllocs()
	for b.Loop() {
		if err := e.WriteContentDelta(w, "the quick brown fox"); err != nil {
			b.Fatal(err)
		}
	}
}
