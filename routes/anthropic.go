// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
)

// messagesRequest is the subset of an Anthropic /v1/messages request this server
// reads. system is a string or a list of text blocks; each message's content is
// a string or a block list, both decoded later by the conversion layer. Sampling
// pointers stay nil when omitted so the sampling cascade can tell unset from zero.
type messagesRequest struct {
	Model         string                 `json:"model"`
	MaxTokens     *int                   `json:"max_tokens"`
	System        json.RawMessage        `json:"system"`
	Messages      []api.AnthropicMessage `json:"messages"`
	Tools         []api.AnthropicTool    `json:"tools"`
	Stream        bool                   `json:"stream"`
	Temperature   *float64               `json:"temperature"`
	TopP          *float64               `json:"top_p"`
	TopK          *int                   `json:"top_k"`
	StopSequences []string               `json:"stop_sequences"`
}

// Messages handles POST /v1/messages (Anthropic Messages API, streaming and
// non-streaming). The mock backend has no chat template to derive native tool
// calling from, so the conversion runs in fallback mode (tools and thinking are
// rendered as text markup) and the model emits plain text.
func (rt *Router) Messages(w http.ResponseWriter, r *http.Request) {
	var req messagesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusUnprocessableEntity, "messages must not be empty", "invalid_request_error", "messages")
		return
	}

	internal, apiTools := api.AnthropicRequestToEngine(req.System, req.Messages, req.Tools, api.AnthropicConvertOptions{})
	msgs := internalToEngineMessages(internal)
	tools := toEngineTools(apiTools)

	prompt, err := rt.eng.BuildPrompt(msgs, tools, engine.PromptOptions{AddGenerationPrompt: true})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "internal_error", "")
		return
	}

	sp := rt.resolveSampling(req.MaxTokens, nil,
		req.Temperature, req.TopP, nil, nil, nil, nil,
		req.TopK, nil, req.StopSequences)

	id := newID("chatcmpl-")
	ereq := &engine.Request{ID: id, Prompt: prompt, Sampling: sp, Arrival: time.Now()}
	ch, err := rt.eng.Submit(ereq)
	if err != nil {
		submitError(w, err)
		return
	}

	model := rt.eng.ModelName()
	if req.Stream {
		rt.streamMessages(w, r, id, model, ch)
		return
	}
	rt.aggregateMessages(w, r, id, model, ch)
}

// aggregateMessages drains the engine output and writes one Anthropic
// MessagesResponse. The body is the reference's pydantic model_dump_json, so it
// is written verbatim rather than through the json encoder.
func (rt *Router) aggregateMessages(w http.ResponseWriter, r *http.Request, id, model string, ch <-chan engine.RequestOutput) {
	var (
		text                  string
		finishReason          = "stop"
		promptTok, completion int
		cached                int
	)
	for o := range ch {
		if o.Err != "" {
			rt.eng.Abort(id)
			writeError(w, http.StatusServiceUnavailable, o.Err, "engine_error", "")
			return
		}
		if o.Finished {
			text = o.OutputText
			finishReason = o.FinishReason
			promptTok = o.PromptTokens
			completion = o.CompletionTokens
			cached = o.CachedTokens
		}
	}
	if r.Context().Err() != nil {
		rt.eng.Abort(id)
		return
	}

	body := api.ConvertInternalToAnthropicResponse(api.AnthropicResponseInput{
		Text:             text,
		Model:            model,
		PromptTokens:     promptTok,
		CompletionTokens: completion,
		FinishReason:     finishReason,
		CachedTokens:     cached,
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(body))
}

// streamMessages streams the Anthropic SSE event sequence for a text-only
// generation: message_start, a single text content block opened with
// content_block_start, text deltas, content_block_stop, then message_delta and
// message_stop. Tool-call streaming is added once a backend emits tool calls.
func (rt *Router) streamMessages(w http.ResponseWriter, r *http.Request, id, model string, ch <-chan engine.RequestOutput) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", "internal_error", "")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	msgID := newID("msg_")
	writeSSE(w, flusher, api.CreateMessageStartEvent(msgID, model, 0))
	writeSSE(w, flusher, api.CreateContentBlockStartEvent(0, "text", "", ""))

	var (
		finishReason          = "stop"
		promptTok, completion int
	)
	for {
		select {
		case <-r.Context().Done():
			rt.eng.Abort(id)
			return
		case o, open := <-ch:
			if !open {
				goto done
			}
			if o.Err != "" {
				writeSSE(w, flusher, api.CreateErrorEvent("engine_error", o.Err))
				return
			}
			if o.NewText != "" {
				writeSSE(w, flusher, api.CreateTextDeltaEvent(0, o.NewText))
			}
			if o.Finished {
				finishReason = o.FinishReason
				promptTok = o.PromptTokens
				completion = o.CompletionTokens
			}
		}
	}
done:
	writeSSE(w, flusher, api.CreateContentBlockStopEvent(0))
	in := promptTok
	writeSSE(w, flusher, api.CreateMessageDeltaEvent(api.MessageDelta{
		StopReason:   api.MapFinishReasonToStopReason(finishReason, false),
		OutputTokens: completion,
		InputTokens:  &in,
	}))
	writeSSE(w, flusher, api.CreateMessageStopEvent())
}

// writeSSE writes one preformatted SSE event and flushes it.
func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string) {
	_, _ = w.Write([]byte(event))
	flusher.Flush()
}

// internalToEngineMessages maps converted internal messages to the prompt
// builder's flat message shape.
func internalToEngineMessages(internal []api.InternalMessage) []engine.Message {
	msgs := make([]engine.Message, len(internal))
	for i, m := range internal {
		msgs[i] = engine.Message{
			Role:             m.Role,
			Content:          m.Content,
			ReasoningContent: m.ReasoningContent,
			ToolCallID:       m.ToolCallID,
		}
	}
	return msgs
}
