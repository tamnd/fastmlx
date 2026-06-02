// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// The fixtures in testdata/parity/anthropic_sse.json are captured from the
// Python reference SSE event layer. The thinking-block signature is the one
// intentional divergence: the reference's opaque placeholder is renamed in the
// fixture so this repo names no other project. Everything else is byte-exact,
// including the ensure_ascii=True escaping of non-ASCII text.

type finishFixture struct {
	FinishReason *string `json:"finish_reason"`
	HasToolCalls bool    `json:"has_tool_calls"`
	Result       *string `json:"result"`
}

type eventFixture struct {
	Label  string `json:"label"`
	Result string `json:"result"`
}

type anthropicSSEFixtures struct {
	Finish []finishFixture `json:"finish"`
	Events []eventFixture  `json:"events"`
}

func loadAnthropicSSEFixtures(t testing.TB) anthropicSSEFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/anthropic_sse.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx anthropicSSEFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Finish) == 0 || len(fx.Events) == 0 {
		t.Fatal("fixtures missing a group")
	}
	return fx
}

func TestMapFinishReasonToStopReasonParity(t *testing.T) {
	for _, fx := range loadAnthropicSSEFixtures(t).Finish {
		fr := ""
		if fx.FinishReason != nil {
			fr = *fx.FinishReason
		}
		want := ""
		if fx.Result != nil {
			want = *fx.Result
		}
		if got := MapFinishReasonToStopReason(fr, fx.HasToolCalls); got != want {
			t.Errorf("MapFinishReasonToStopReason(%q, %v) = %q, want %q",
				fr, fx.HasToolCalls, got, want)
		}
	}
}

// eventByLabel reproduces the exact arguments the capture script used, so each
// Go constructor is checked against its reference output.
func eventByLabel(label string) (string, bool) {
	intp := func(i int) *int { return &i }
	switch label {
	case "message_start":
		return CreateMessageStartEvent("msg_abc", "claude-test", 12), true
	case "message_start_zero":
		return CreateMessageStartEvent("msg_1", "m", 0), true
	case "cbs_text":
		return CreateContentBlockStartEvent(0, "text", "", ""), true
	case "cbs_tool_use":
		return CreateContentBlockStartEvent(1, "tool_use", "toolu_x", "get_weather"), true
	case "cbs_tool_use_defaults":
		return CreateContentBlockStartEvent(2, "tool_use", "", ""), true
	case "cbs_thinking":
		return CreateContentBlockStartEvent(0, "thinking", "", ""), true
	case "cbs_other":
		return CreateContentBlockStartEvent(3, "image", "", ""), true
	case "thinking_delta":
		return CreateThinkingDeltaEvent(0, "let me think"), true
	case "text_delta":
		return CreateTextDeltaEvent(0, "Hello, world"), true
	case "text_delta_unicode":
		return CreateTextDeltaEvent(0, "café 日本語 ☃"), true
	case "text_delta_escapes":
		return CreateTextDeltaEvent(1, "line1\nline2\t\"q\"\\"), true
	case "input_json_delta":
		return CreateInputJSONDeltaEvent(1, `{"city": "Paris"}`), true
	case "cb_stop":
		return CreateContentBlockStopEvent(2), true
	case "msg_delta_basic":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "end_turn", OutputTokens: 42}), true
	case "msg_delta_seq":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "stop_sequence", OutputTokens: 5, StopSequence: "STOP"}), true
	case "msg_delta_input":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "end_turn", OutputTokens: 10, InputTokens: intp(100)}), true
	case "msg_delta_cache":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "end_turn", OutputTokens: 10, InputTokens: intp(100), CachedTokens: 30, RequestUsesCacheControl: true}), true
	case "msg_delta_cache_over":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "max_tokens", OutputTokens: 7, InputTokens: intp(50), CachedTokens: 80, RequestUsesCacheControl: true}), true
	case "msg_delta_none":
		return CreateMessageDeltaEvent(MessageDelta{StopReason: "", OutputTokens: 0}), true
	case "msg_stop":
		return CreateMessageStopEvent(), true
	case "ping":
		return CreatePingEvent(), true
	case "error":
		return CreateErrorEvent("invalid_request_error", "bad input"), true
	case "error_unicode":
		return CreateErrorEvent("overloaded_error", "trop de requêtes"), true
	}
	return "", false
}

func TestAnthropicSSEEventParity(t *testing.T) {
	for _, fx := range loadAnthropicSSEFixtures(t).Events {
		t.Run(fx.Label, func(t *testing.T) {
			got, ok := eventByLabel(fx.Label)
			if !ok {
				t.Fatalf("no Go constructor wired for label %q", fx.Label)
			}
			if got != fx.Result {
				t.Errorf("event %q\n got  %q\n want %q", fx.Label, got, fx.Result)
			}
		})
	}
}

func BenchmarkCreateTextDeltaEvent(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = CreateTextDeltaEvent(0, "the quick brown fox jumps over the lazy dog")
	}
}
