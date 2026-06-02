// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
)

// responsesRequest is the subset of a Responses /v1/responses request this
// server reads. input is a string or an item list; instructions is the system
// prompt; tools are raw Responses tool definitions echoed back in the response.
// Sampling pointers stay nil when omitted so the cascade can tell unset from
// zero. Streaming is not yet wired (the Responses SSE event set is its own
// layer), so a streamed request is reported as unsupported.
type responsesRequest struct {
	Model              string            `json:"model"`
	Input              json.RawMessage   `json:"input"`
	Instructions       string            `json:"instructions"`
	Tools              []json.RawMessage `json:"tools"`
	ToolChoice         json.RawMessage   `json:"tool_choice"`
	Stream             bool              `json:"stream"`
	Temperature        json.RawMessage   `json:"temperature"`
	TopP               json.RawMessage   `json:"top_p"`
	MaxOutputTokens    json.RawMessage   `json:"max_output_tokens"`
	PreviousResponseID string            `json:"previous_response_id"`
}

// rawInt returns the int value of a raw JSON number, or nil when absent or
// unparseable. The Responses sampling cascade uses it for max_output_tokens.
func rawInt(raw json.RawMessage) *int {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return nil
	}
	return &n
}

// rawFloat returns the float value of a raw JSON number, or nil when absent or
// unparseable.
func rawFloat(raw json.RawMessage) *float64 {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil
	}
	return &f
}

// Responses handles POST /v1/responses (OpenAI Responses API, non-streaming).
// The request input is converted to internal messages, the mock backend
// generates text, and the result is assembled into a ResponseObject whose wire
// form matches the reference's pydantic model_dump_json.
func (rt *Router) Responses(w http.ResponseWriter, r *http.Request) {
	var req responsesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}
	if len(req.Input) == 0 || string(req.Input) == "null" {
		writeError(w, http.StatusUnprocessableEntity, "input must not be empty", "invalid_request_error", "input")
		return
	}

	internal, apiTools := api.ResponsesRequestToEngine(req.Input, req.Instructions, req.Tools)
	msgs := internalToEngineMessages(internal)
	tools := toEngineTools(apiTools)

	prompt, err := rt.eng.BuildPrompt(msgs, tools, engine.PromptOptions{AddGenerationPrompt: true})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "internal_error", "")
		return
	}

	sp := rt.resolveSampling(nil, rawInt(req.MaxOutputTokens),
		rawFloat(req.Temperature), rawFloat(req.TopP), nil, nil, nil, nil,
		nil, nil, nil)

	id := newID("chatcmpl-")
	ereq := &engine.Request{ID: id, Prompt: prompt, Sampling: sp, Arrival: time.Now()}
	ch, err := rt.eng.Submit(ereq)
	if err != nil {
		submitError(w, err)
		return
	}

	if req.Stream {
		rt.streamResponses(w, r, id, req, ch)
		return
	}

	var (
		text                  string
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
			promptTok = o.PromptTokens
			completion = o.CompletionTokens
			cached = o.CachedTokens
		}
	}
	if r.Context().Err() != nil {
		rt.eng.Abort(id)
		return
	}

	body := api.BuildResponsesResponse(int(time.Now().Unix()), api.ResponsesResult{
		Model:              rt.eng.ModelName(),
		Text:               text,
		PromptTokens:       promptTok,
		CompletionTokens:   completion,
		CachedTokens:       cached,
		Temperature:        req.Temperature,
		TopP:               req.TopP,
		MaxOutputTokens:    req.MaxOutputTokens,
		ToolChoice:         req.ToolChoice,
		Tools:              req.Tools,
		PreviousResponseID: req.PreviousResponseID,
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(body))
}

// streamResponses streams the Responses SSE event sequence for a text-only
// generation: response.created and response.in_progress, the message item opened
// with output_item.added and content_part.added, output_text.delta per chunk,
// then output_text.done, content_part.done, output_item.done, and
// response.completed with final usage. Reasoning and function-call events land
// when a backend emits them.
func (rt *Router) streamResponses(w http.ResponseWriter, r *http.Request, id string, req responsesRequest, ch <-chan engine.RequestOutput) {
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

	sw := api.NewResponsesStreamWriter(api.ResponsesStreamInit{
		ResponseID:         newID("resp_"),
		MessageID:          newID("msg_"),
		Model:              rt.eng.ModelName(),
		CreatedAt:          int(time.Now().Unix()),
		Temperature:        req.Temperature,
		TopP:               req.TopP,
		MaxOutputTokens:    req.MaxOutputTokens,
		ToolChoice:         req.ToolChoice,
		Tools:              req.Tools,
		PreviousResponseID: req.PreviousResponseID,
	})
	for _, ev := range sw.Start() {
		writeSSE(w, flusher, ev)
	}

	var (
		text                  strings.Builder
		promptTok, completion int
		cached                int
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
				writeSSE(w, flusher, sw.Failed())
				return
			}
			if o.NewText != "" {
				text.WriteString(o.NewText)
				writeSSE(w, flusher, sw.TextDelta(o.NewText))
			}
			if o.Finished {
				promptTok = o.PromptTokens
				completion = o.CompletionTokens
				cached = o.CachedTokens
			}
		}
	}
done:
	for _, ev := range sw.Finish(text.String(), promptTok, completion, cached) {
		writeSSE(w, flusher, ev)
	}
}
