// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// parseHarmonyRequest pulls the system value and message list out of a captured
// request object (an ordered jval) into the call shape the converter takes.
func parseHarmonyRequest(t *testing.T, raw string) (jval, []AnthropicInMessage) {
	t.Helper()
	req, ok := parseOrdered(raw)
	if !ok || req.kind != kindObject {
		t.Fatalf("request is not a JSON object: %s", raw)
	}
	system, _ := req.getField("system")
	msgsVal, _ := req.getField("messages")
	var messages []AnthropicInMessage
	for _, m := range msgsVal.arr {
		content, _ := m.getField("content")
		messages = append(messages, AnthropicInMessage{Role: m.getString("role"), Content: content})
	}
	return system, messages
}

func TestConvertAnthropicToInternalHarmonyParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "convert_anthropic_harmony.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Request json.RawMessage `json:"request"`
		Result  json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		system, messages := parseHarmonyRequest(t, string(c.Request))
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: result is not valid JSON: %s", i, c.Result)
		}
		got := jval{kind: kindArray, arr: ConvertAnthropicToInternalHarmony(system, messages)}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d:\n got %s\nwant %s", i, g, w)
		}
	}
}

func BenchmarkConvertAnthropicToInternalHarmony(b *testing.B) {
	b.ReportAllocs()
	req, _ := parseOrdered(`{"system":"be brief","messages":[` +
		`{"role":"user","content":"weather?"},` +
		`{"role":"assistant","content":[{"type":"text","text":"checking"},` +
		`{"type":"tool_use","id":"tu1","name":"get_weather","input":{"city":"Tokyo"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"{\"temp\": 21}"}]}]}`)
	system, _ := req.getField("system")
	msgsVal, _ := req.getField("messages")
	var messages []AnthropicInMessage
	for _, m := range msgsVal.arr {
		content, _ := m.getField("content")
		messages = append(messages, AnthropicInMessage{Role: m.getString("role"), Content: content})
	}
	for b.Loop() {
		_ = ConvertAnthropicToInternalHarmony(system, messages)
	}
}
