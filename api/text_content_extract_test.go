// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// messagesFromRaw decodes a captured message list (an ordered JSON array) into
// the []jval the extractor takes.
func messagesFromRaw(t *testing.T, raw json.RawMessage) []jval {
	t.Helper()
	v, ok := parseOrdered(string(raw))
	if !ok || v.kind != kindArray {
		t.Fatalf("messages is not a JSON array: %s", raw)
	}
	return v.arr
}

func TestExtractTextContentParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "extract_text_content.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		NativeToolCalling      bool            `json:"native_tool_calling"`
		NativeReasoningContent bool            `json:"native_reasoning_content"`
		Messages               json.RawMessage `json:"messages"`
		Result                 json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		messages := messagesFromRaw(t, c.Messages)
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: result is not valid JSON: %s", i, c.Result)
		}
		got := jval{kind: kindArray, arr: ExtractTextContent(messages, c.NativeToolCalling, c.NativeReasoningContent)}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d:\n got %s\nwant %s", i, g, w)
		}
	}
}

func BenchmarkExtractTextContent(b *testing.B) {
	b.ReportAllocs()
	v, _ := parseOrdered(`[` +
		`{"role":"system","content":"be brief"},` +
		`{"role":"user","content":[{"type":"text","text":"weather?"}]},` +
		`{"role":"assistant","content":"","reasoning_content":"need a tool","tool_calls":[` +
		`{"id":"t1","function":{"name":"get_weather","arguments":"{\"city\": \"Tokyo\"}"}}]},` +
		`{"role":"tool","tool_call_id":"t1","content":"{\"temp\": 21}"},` +
		`{"role":"user","content":"thanks"}]`)
	messages := v.arr
	for b.Loop() {
		_ = ExtractTextContent(messages, true, true)
	}
}
