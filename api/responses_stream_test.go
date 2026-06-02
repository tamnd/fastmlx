// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// Fixtures in testdata/parity/responses_stream.json are the exact SSE event
// strings the reference streaming handler emits, captured by reconstructing each
// event payload as that handler builds it and serializing with the same
// json.dumps the reference uses. These are contractual wire format, so the
// comparison is byte-exact (the full event string, event line and data line and
// trailing blank line included). Sentinel ids pin the random values:
// resp_PARITY, msg_PARITY, rs_PARITY, fc_PARITY0, created_at 0.

const (
	streamRespID  = "resp_PARITY"
	streamMsgID   = "msg_PARITY"
	streamReasID  = "rs_PARITY"
	streamFCID    = "fc_PARITY0"
	streamCreated = 0
)

type responsesStreamFixtures struct {
	Cases []struct {
		Label string `json:"label"`
		Raw   string `json:"raw"`
	} `json:"cases"`
}

func loadResponsesStreamFixtures(t testing.TB) responsesStreamFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/responses_stream.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx responsesStreamFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	return fx
}

// buildStreamEvent reconstructs the event for a fixture label using the same
// constructors and arguments the reference handler would. A label with no case
// returns ("", false).
func buildStreamEvent(label string) (string, bool) {
	base := streamResponseInit{ID: streamRespID, Model: "mock-model", CreatedAt: streamCreated, Status: "in_progress"}
	withSampling := base
	withSampling.Temperature = jnumLit("0.7")
	withSampling.TopP = jnumLit("0.9")
	withSampling.MaxOutputTokens = jnumLit("256")
	withSampling.PreviousResponseID = "resp_PREV"

	msgItem := messageItemCompleted(streamMsgID, "Hello world")
	usage := buildStreamUsage(11, 2, 0, 0)

	completedFinal := streamResponseInit{ID: streamRespID, Model: "mock-model", CreatedAt: streamCreated}
	completedFinalSampling := completedFinal
	completedFinalSampling.Temperature = jnumLit("0.5")
	completedFinalSampling.TopP = jnumLit("0.8")
	completedFinalSampling.MaxOutputTokens = jnumLit("128")
	completedFinalSampling.PreviousResponseID = "resp_PREV"

	switch label {
	case "created_minimal":
		return evCreated(1, buildStreamInitial(base)), true
	case "created_with_sampling":
		return evCreated(1, buildStreamInitial(withSampling)), true
	case "in_progress":
		return evInProgress(2, buildStreamInitial(base)), true
	case "output_item_added_reasoning":
		return evOutputItemAdded(3, 0, reasoningItem(streamReasID, "in_progress", "", false)), true
	case "reasoning_summary_part_added":
		return evReasoningSummaryPartAdded(4, streamReasID, 0, 0, summaryTextPart("")), true
	case "reasoning_summary_text_delta":
		return evReasoningSummaryTextDelta(5, streamReasID, 0, 0, "thinking"), true
	case "reasoning_summary_text_done":
		return evReasoningSummaryTextDone(6, streamReasID, 0, 0, "thinking done"), true
	case "reasoning_summary_part_done":
		return evReasoningSummaryPartDone(7, streamReasID, 0, 0, summaryTextPart("thinking done")), true
	case "output_item_done_reasoning":
		return evOutputItemDone(8, 0, reasoningItem(streamReasID, "completed", "thinking done", true)), true
	case "output_item_added_message":
		return evOutputItemAdded(3, 0, messageItemInProgress(streamMsgID)), true
	case "content_part_added":
		return evContentPartAdded(4, streamMsgID, 0, 0, outputTextPart("")), true
	case "output_text_delta":
		return evOutputTextDelta(5, streamMsgID, 0, 0, "Hello"), true
	case "output_text_delta_unicode":
		return evOutputTextDelta(5, streamMsgID, 0, 0, "café 你好"), true
	case "output_text_done":
		return evOutputTextDone(6, streamMsgID, 0, 0, "Hello world"), true
	case "content_part_done":
		return evContentPartDone(7, streamMsgID, 0, 0, outputTextPart("Hello world")), true
	case "output_item_done_message":
		return evOutputItemDone(8, 0, msgItem), true
	case "output_item_added_function_call":
		return evOutputItemAdded(9, 1, functionCallItem(streamFCID, "call_abc", "get_weather", "", "in_progress")), true
	case "function_call_arguments_delta":
		return evFunctionCallArgsDelta(10, streamFCID, 1, `{"city": "Paris"}`), true
	case "function_call_arguments_done":
		return evFunctionCallArgsDone(11, streamFCID, 1, `{"city": "Paris"}`), true
	case "output_item_done_function_call":
		return evOutputItemDone(12, 1, functionCallItem(streamFCID, "call_abc", "get_weather", `{"city": "Paris"}`, "completed")), true
	case "completed_message_only":
		return evCompleted(9, buildStreamFinal(completedFinal, []jval{msgItem}, usage)), true
	case "completed_no_usage":
		return evCompleted(9, buildStreamFinal(completedFinalSampling, []jval{msgItem}, jnull())), true
	case "failed":
		f := base
		f.Status = "failed"
		return evFailed(4, buildStreamInitial(f)), true
	default:
		return "", false
	}
}

// jnumLit builds a number jval from its exact textual form.
func jnumLit(s string) jval { return jval{kind: kindNumber, s: s} }

func TestResponsesStreamEventParity(t *testing.T) {
	fx := loadResponsesStreamFixtures(t)
	if len(fx.Cases) == 0 {
		t.Fatal("no fixtures loaded")
	}
	for _, c := range fx.Cases {
		t.Run(c.Label, func(t *testing.T) {
			got, ok := buildStreamEvent(c.Label)
			if !ok {
				t.Fatalf("no constructor mapped for label %q", c.Label)
			}
			if got != c.Raw {
				t.Errorf("event mismatch for %q\n got  %q\n want %q", c.Label, got, c.Raw)
			}
		})
	}
}

func BenchmarkResponsesStreamEvents(b *testing.B) {
	base := streamResponseInit{ID: streamRespID, Model: "mock-model", CreatedAt: streamCreated, Status: "in_progress"}
	msgItem := messageItemCompleted(streamMsgID, "Hello world")
	usage := buildStreamUsage(11, 2, 0, 0)
	final := streamResponseInit{ID: streamRespID, Model: "mock-model", CreatedAt: streamCreated}
	b.ReportAllocs()
	for b.Loop() {
		_ = evCreated(1, buildStreamInitial(base))
		_ = evOutputTextDelta(5, streamMsgID, 0, 0, "Hello")
		_ = evCompleted(9, buildStreamFinal(final, []jval{msgItem}, usage))
	}
}
