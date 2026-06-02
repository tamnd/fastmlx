// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
)

// Completions handles POST /v1/completions (legacy text completions).
func (rt *Router) Completions(w http.ResponseWriter, r *http.Request) {
	var req api.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}
	prompt := req.Prompt.First()
	if prompt == "" {
		writeError(w, http.StatusUnprocessableEntity, "prompt must not be empty", "invalid_request_error", "prompt")
		return
	}

	sp := rt.resolveSampling(req.MaxTokens, nil, req.Temperature, req.TopP, nil, nil, nil, nil,
		req.TopK, req.Seed, []string(req.Stop))

	id := newID("cmpl-")
	ereq := &engine.Request{ID: id, Prompt: prompt, Sampling: sp, Arrival: time.Now()}
	ch, err := rt.eng.Submit(ereq)
	if err != nil {
		submitError(w, err)
		return
	}

	created := time.Now().Unix()
	model := rt.eng.ModelName()

	if req.Stream {
		rt.streamCompletion(w, r, id, model, created, ch)
		return
	}

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
	writeJSON(w, http.StatusOK, api.CompletionResponse{
		ID: id, Object: "text_completion", Created: created, Model: model,
		Choices: []api.CompletionChoice{{Text: text, Index: 0, FinishReason: finishReason}},
		Usage:   *makeUsage(promptTok, completion, cached),
	})
}

func (rt *Router) streamCompletion(w http.ResponseWriter, r *http.Request, id, model string, created int64, ch <-chan engine.RequestOutput) {
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

	finishReason := "stop"
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
				finishReason = "error"
				goto done
			}
			if o.NewText != "" {
				writeCompletionChunk(w, id, model, created, o.NewText, nil)
				flusher.Flush()
			}
			if o.Finished {
				finishReason = o.FinishReason
			}
		}
	}
done:
	writeCompletionChunk(w, id, model, created, "", &finishReason)
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

func writeCompletionChunk(w http.ResponseWriter, id, model string, created int64, text string, finishReason *string) {
	chunk := api.CompletionResponse{
		ID: id, Object: "text_completion", Created: created, Model: model,
		Choices: []api.CompletionChoice{{Text: text, Index: 0}},
	}
	if finishReason != nil {
		chunk.Choices[0].FinishReason = *finishReason
	}
	b, _ := json.Marshal(chunk)
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write(b)
	_, _ = w.Write([]byte("\n\n"))
}
