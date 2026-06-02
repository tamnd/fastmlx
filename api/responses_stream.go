// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// This file ports the Responses API streaming SSE event layer. The reference
// builds these events inline in its streaming handler, each rendered with
// json.dumps over a plain dict, so the wire form is the same ensure_ascii=True,
// ", "/": "-separated encoding that dumpASCII already produces. The event
// vocabulary is the full set the handler can emit (reasoning, message, and
// function_call items); the text-only path the mock backend drives uses the
// ResponsesStreamWriter below.
//
// Two response-object shapes appear inside events. The created/in_progress/
// failed events embed the initial object built from the response model with
// exclude_none, so null fields are dropped and metadata stays as {}. The
// completed event embeds a hand-built dict, so its temperature/top_p/
// max_output_tokens/usage keys are always present (null when unset) and there is
// no metadata key. Both are reproduced exactly here.

// formatResponsesSSE renders one Responses SSE event: an "event:" line, a
// "data:" line with the json.dumps payload, and a blank separator.
func formatResponsesSSE(eventType string, payload jval) string {
	var b strings.Builder
	b.WriteString("event: ")
	b.WriteString(eventType)
	b.WriteString("\ndata: ")
	b.WriteString(payload.dumpASCII())
	b.WriteString("\n\n")
	return b.String()
}

// orNull returns v, or JSON null when v is the zero (unset) jval.
func orNull(v jval) jval {
	if v.kind == 0 {
		return jnull()
	}
	return v
}

// streamResponseInit carries the request echo fields the embedded response
// objects repeat. Temperature, TopP, MaxOutputTokens, and ToolChoice are jval so
// an unset field (the zero jval) can be told from an explicit value; pass parsed
// request literals so numbers round-trip without reformatting.
type streamResponseInit struct {
	ID                 string
	Model              string
	CreatedAt          int
	Status             string
	Temperature        jval
	TopP               jval
	MaxOutputTokens    jval
	Tools              []jval
	ToolChoice         jval
	PreviousResponseID string
}

// buildStreamInitial builds the initial in-progress (or failed) response object
// as the reference's model_dump(exclude_none=True): unset sampling echoes are
// dropped, and metadata is an empty object.
func buildStreamInitial(in streamResponseInit) jval {
	o := jval{kind: kindObject}
	add := func(k string, v jval) { o.obj = append(o.obj, jkv{k, v}) }
	add("id", jstr(in.ID))
	add("object", jstr("response"))
	add("created_at", jint(in.CreatedAt))
	add("model", jstr(in.Model))
	add("status", jstr(in.Status))
	add("output", jval{kind: kindArray})
	toolChoice := in.ToolChoice
	if toolChoice.kind == 0 {
		toolChoice = jstr("auto")
	}
	add("tool_choice", toolChoice)
	tools := jval{kind: kindArray}
	if len(in.Tools) > 0 {
		tools.arr = in.Tools
	}
	add("tools", tools)
	if in.Temperature.kind != 0 {
		add("temperature", in.Temperature)
	}
	if in.TopP.kind != 0 {
		add("top_p", in.TopP)
	}
	if in.MaxOutputTokens.kind != 0 {
		add("max_output_tokens", in.MaxOutputTokens)
	}
	if in.PreviousResponseID != "" {
		add("previous_response_id", jstr(in.PreviousResponseID))
	}
	add("metadata", jval{kind: kindObject})
	return o
}

// buildStreamFinal builds the completed response object as the reference's
// hand-built dict: usage and the sampling echoes are always present (null when
// unset), and there is no metadata key.
func buildStreamFinal(in streamResponseInit, output []jval, usage jval) jval {
	o := jval{kind: kindObject}
	add := func(k string, v jval) { o.obj = append(o.obj, jkv{k, v}) }
	add("id", jstr(in.ID))
	add("object", jstr("response"))
	add("created_at", jint(in.CreatedAt))
	add("model", jstr(in.Model))
	add("status", jstr("completed"))
	add("output", jval{kind: kindArray, arr: output})
	add("usage", orNull(usage))
	toolChoice := in.ToolChoice
	if toolChoice.kind == 0 {
		toolChoice = jstr("auto")
	}
	add("tool_choice", toolChoice)
	tools := jval{kind: kindArray}
	if len(in.Tools) > 0 {
		tools.arr = in.Tools
	}
	add("tools", tools)
	add("temperature", orNull(in.Temperature))
	add("top_p", orNull(in.TopP))
	add("max_output_tokens", orNull(in.MaxOutputTokens))
	if in.PreviousResponseID != "" {
		add("previous_response_id", jstr(in.PreviousResponseID))
	}
	return o
}

// buildStreamUsage builds the usage object embedded in the completed event.
func buildStreamUsage(input, output, cached, reasoning int) jval {
	return jobj(
		"input_tokens", jint(input),
		"output_tokens", jint(output),
		"total_tokens", jint(input+output),
		"input_tokens_details", jobj("cached_tokens", jint(cached)),
		"output_tokens_details", jobj("reasoning_tokens", jint(reasoning)),
	)
}

// --- output item builders ---

func summaryTextPart(text string) jval {
	return jobj("type", jstr("summary_text"), "text", jstr(text))
}

func outputTextPart(text string) jval {
	return jobj("type", jstr("output_text"), "text", jstr(text), "annotations", jval{kind: kindArray})
}

// reasoningItem builds a reasoning output item. A completed item carries one
// summary_text part; an in-progress item carries an empty summary.
func reasoningItem(id, status string, summaryText string, withSummary bool) jval {
	summary := jval{kind: kindArray}
	if withSummary {
		summary.arr = []jval{summaryTextPart(summaryText)}
	}
	return jobj("type", jstr("reasoning"), "id", jstr(id), "status", jstr(status), "summary", summary)
}

func messageItemInProgress(id string) jval {
	return jobj("type", jstr("message"), "id", jstr(id), "status", jstr("in_progress"),
		"role", jstr("assistant"), "content", jval{kind: kindArray})
}

func messageItemCompleted(id, text string) jval {
	return jobj("type", jstr("message"), "id", jstr(id), "status", jstr("completed"),
		"role", jstr("assistant"), "content", jval{kind: kindArray, arr: []jval{outputTextPart(text)}})
}

func functionCallItem(id, callID, name, arguments, status string) jval {
	return jobj("type", jstr("function_call"), "id", jstr(id), "call_id", jstr(callID),
		"name", jstr(name), "arguments", jstr(arguments), "status", jstr(status))
}

// --- event constructors ---

func evCreated(seq int, response jval) string {
	return formatResponsesSSE("response.created", jobj(
		"type", jstr("response.created"), "response", response, "sequence_number", jint(seq)))
}

func evInProgress(seq int, response jval) string {
	return formatResponsesSSE("response.in_progress", jobj(
		"type", jstr("response.in_progress"), "response", response, "sequence_number", jint(seq)))
}

func evFailed(seq int, response jval) string {
	return formatResponsesSSE("response.failed", jobj(
		"type", jstr("response.failed"), "response", response, "sequence_number", jint(seq)))
}

func evCompleted(seq int, response jval) string {
	return formatResponsesSSE("response.completed", jobj(
		"type", jstr("response.completed"), "response", response, "sequence_number", jint(seq)))
}

func evOutputItemAdded(seq, outputIndex int, item jval) string {
	return formatResponsesSSE("response.output_item.added", jobj(
		"type", jstr("response.output_item.added"),
		"output_index", jint(outputIndex), "item", item, "sequence_number", jint(seq)))
}

func evOutputItemDone(seq, outputIndex int, item jval) string {
	return formatResponsesSSE("response.output_item.done", jobj(
		"type", jstr("response.output_item.done"),
		"output_index", jint(outputIndex), "item", item, "sequence_number", jint(seq)))
}

func evReasoningSummaryPartAdded(seq int, itemID string, outputIndex, summaryIndex int, part jval) string {
	return formatResponsesSSE("response.reasoning_summary_part.added", jobj(
		"type", jstr("response.reasoning_summary_part.added"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"summary_index", jint(summaryIndex), "part", part, "sequence_number", jint(seq)))
}

func evReasoningSummaryTextDelta(seq int, itemID string, outputIndex, summaryIndex int, delta string) string {
	return formatResponsesSSE("response.reasoning_summary_text.delta", jobj(
		"type", jstr("response.reasoning_summary_text.delta"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"summary_index", jint(summaryIndex), "delta", jstr(delta), "sequence_number", jint(seq)))
}

func evReasoningSummaryTextDone(seq int, itemID string, outputIndex, summaryIndex int, text string) string {
	return formatResponsesSSE("response.reasoning_summary_text.done", jobj(
		"type", jstr("response.reasoning_summary_text.done"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"summary_index", jint(summaryIndex), "text", jstr(text), "sequence_number", jint(seq)))
}

func evReasoningSummaryPartDone(seq int, itemID string, outputIndex, summaryIndex int, part jval) string {
	return formatResponsesSSE("response.reasoning_summary_part.done", jobj(
		"type", jstr("response.reasoning_summary_part.done"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"summary_index", jint(summaryIndex), "part", part, "sequence_number", jint(seq)))
}

func evContentPartAdded(seq int, itemID string, outputIndex, contentIndex int, part jval) string {
	return formatResponsesSSE("response.content_part.added", jobj(
		"type", jstr("response.content_part.added"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"content_index", jint(contentIndex), "part", part, "sequence_number", jint(seq)))
}

func evContentPartDone(seq int, itemID string, outputIndex, contentIndex int, part jval) string {
	return formatResponsesSSE("response.content_part.done", jobj(
		"type", jstr("response.content_part.done"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"content_index", jint(contentIndex), "part", part, "sequence_number", jint(seq)))
}

func evOutputTextDelta(seq int, itemID string, outputIndex, contentIndex int, delta string) string {
	return formatResponsesSSE("response.output_text.delta", jobj(
		"type", jstr("response.output_text.delta"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"content_index", jint(contentIndex), "delta", jstr(delta), "sequence_number", jint(seq)))
}

func evOutputTextDone(seq int, itemID string, outputIndex, contentIndex int, text string) string {
	return formatResponsesSSE("response.output_text.done", jobj(
		"type", jstr("response.output_text.done"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"content_index", jint(contentIndex), "text", jstr(text), "sequence_number", jint(seq)))
}

func evFunctionCallArgsDelta(seq int, itemID string, outputIndex int, delta string) string {
	return formatResponsesSSE("response.function_call_arguments.delta", jobj(
		"type", jstr("response.function_call_arguments.delta"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"delta", jstr(delta), "sequence_number", jint(seq)))
}

func evFunctionCallArgsDone(seq int, itemID string, outputIndex int, arguments string) string {
	return formatResponsesSSE("response.function_call_arguments.done", jobj(
		"type", jstr("response.function_call_arguments.done"),
		"item_id", jstr(itemID), "output_index", jint(outputIndex),
		"arguments", jstr(arguments), "sequence_number", jint(seq)))
}
