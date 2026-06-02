// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
)

// ChatCompletions handles POST /v1/chat/completions (streaming + non-streaming).
func (rt *Router) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req api.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusUnprocessableEntity, "messages must not be empty", "invalid_request_error", "messages")
		return
	}

	msgs := make([]engine.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = engine.Message{
			Role:             m.Role,
			Content:          m.ContentText(),
			Name:             m.Name,
			ToolCallID:       m.ToolCallID,
			ReasoningContent: m.ReasoningContent,
		}
	}
	tools := toEngineTools(req.Tools)
	prompt, err := rt.eng.BuildPrompt(msgs, tools, engine.PromptOptions{AddGenerationPrompt: true})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "internal_error", "")
		return
	}

	sp := rt.resolveSampling(req.MaxTokens, req.MaxCompletionTokens,
		req.Temperature, req.TopP, req.MinP, req.RepetitionPenalty, req.PresencePenalty, req.FrequencyPenalty,
		req.TopK, req.Seed, []string(req.Stop))

	id := newID("chatcmpl-")
	ereq := &engine.Request{ID: id, Prompt: prompt, Sampling: sp, Arrival: time.Now()}
	ch, err := rt.eng.Submit(ereq)
	if err != nil {
		submitError(w, err)
		return
	}

	created := time.Now().Unix()
	model := rt.eng.ModelName()
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	if req.Stream {
		rt.streamChat(w, r, id, model, created, includeUsage, ch)
		return
	}
	rt.aggregateChat(w, r, id, model, created, ch)
}

func (rt *Router) streamChat(w http.ResponseWriter, r *http.Request, id, model string, created int64, includeUsage bool, ch <-chan engine.RequestOutput) {
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

	enc := api.NewChunkEncoder(id, model, created)
	_ = enc.WriteRole(w)
	flusher.Flush()

	var (
		finishReason          = "stop"
		promptTok, completion int
		cached                int
	)
	for {
		select {
		case <-r.Context().Done():
			// Client hung up: cancel the engine request and release its slot.
			rt.eng.Abort(id)
			return
		case o, open := <-ch:
			if !open {
				goto done
			}
			if o.Err != "" {
				finishReason = "error"
				goto done
			}
			if o.NewText != "" {
				_ = enc.WriteContentDelta(w, o.NewText)
				flusher.Flush()
			}
			if o.Finished {
				finishReason = o.FinishReason
				promptTok = o.PromptTokens
				completion = o.CompletionTokens
				cached = o.CachedTokens
			}
		}
	}
done:
	var usage *api.Usage
	if includeUsage {
		usage = makeUsage(promptTok, completion, cached)
	}
	_ = enc.WriteFinish(w, finishReason, usage)
	_ = enc.WriteDone(w)
	flusher.Flush()
}

func (rt *Router) aggregateChat(w http.ResponseWriter, r *http.Request, id, model string, created int64, ch <-chan engine.RequestOutput) {
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
	// Honor client disconnect on the non-streaming path too.
	if r.Context().Err() != nil {
		rt.eng.Abort(id)
		return
	}
	resp := api.ChatCompletionResponse{
		ID: id, Object: "chat.completion", Created: created, Model: model,
		Choices: []api.ChatChoice{{
			Index:        0,
			Message:      api.ResponseMessage{Role: "assistant", Content: text},
			FinishReason: finishReason,
		}},
		Usage: *makeUsage(promptTok, completion, cached),
	}
	writeJSON(w, http.StatusOK, resp)
}

func makeUsage(prompt, completion, cached int) *api.Usage {
	u := &api.Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
	}
	if cached > 0 {
		u.CacheReadInputTokens = cached
		u.PromptTokensDetails = &api.PromptTokensDetails{CachedTokens: cached}
	}
	return u
}

func toEngineTools(tools []api.Tool) []engine.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]engine.Tool, len(tools))
	for i, t := range tools {
		out[i] = engine.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	return out
}

func submitError(w http.ResponseWriter, err error) {
	if errors.Is(err, engine.ErrQueueFull) {
		writeError(w, http.StatusServiceUnavailable, "server is at capacity, retry later", "server_overloaded", "")
		return
	}
	writeError(w, http.StatusInternalServerError, err.Error(), "internal_error", "")
}
