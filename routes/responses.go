// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
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
	if req.Stream {
		writeError(w, http.StatusNotImplemented,
			"streaming for /v1/responses lands in a later milestone", "not_implemented_error", "stream")
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
