// SPDX-License-Identifier: MIT OR Apache-2.0

// Package routes implements the OpenAI-compatible HTTP handlers. It depends on a
// narrow Engine interface so any concrete engine (the mock-backed BatchedEngine
// today, the compute-backed one later) serves the same routes unchanged. Maps
// the OpenAI-compatible HTTP surface.
package routes

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"time"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/tokenizer"
)

// Engine is the subset of the engine surface the routes need. enginecore's
// BatchedEngine satisfies it.
type Engine interface {
	ModelName() string
	Tokenizer() tokenizer.Tokenizer
	CountTokens(text string) int
	BuildPrompt(msgs []engine.Message, tools []engine.Tool, opts engine.PromptOptions) (string, error)
	Submit(req *engine.Request) (<-chan engine.RequestOutput, error)
	Abort(id string)
	Defaults() engine.SamplingParams
	InFlight() int
}

// Router holds the engine and bind-time state shared by the handlers.
type Router struct {
	eng     Engine
	started time.Time
	mcp     mcpManager // nil until an MCP manager is attached
}

// New builds a router over an engine.
func New(eng Engine) *Router {
	return &Router{eng: eng, started: time.Now()}
}

// Register mounts the OpenAI routes on a ServeMux using Go 1.22 method+path
// patterns.
func (rt *Router) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", rt.Health)
	mux.HandleFunc("GET /api/status", rt.Status)
	mux.HandleFunc("GET /v1/models", rt.ListModels)
	mux.HandleFunc("POST /v1/chat/completions", rt.ChatCompletions)
	mux.HandleFunc("POST /v1/completions", rt.Completions)
	mux.HandleFunc("POST /v1/embeddings", rt.Embeddings)
	mux.HandleFunc("POST /v1/messages", rt.Messages)
	mux.HandleFunc("POST /v1/messages/count_tokens", rt.CountTokens)
	mux.HandleFunc("POST /v1/responses", rt.Responses)
	mux.HandleFunc("POST /v1/rerank", rt.Rerank)
	mux.HandleFunc("GET /v1/mcp/tools", rt.MCPTools)
	mux.HandleFunc("GET /v1/mcp/servers", rt.MCPServers)
	mux.HandleFunc("POST /v1/mcp/execute", rt.MCPExecute)
}

// Health reports liveness.
func (rt *Router) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// Status reports the loaded model, in-flight count, and uptime.
func (rt *Router) Status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":         "ok",
		"model":          rt.eng.ModelName(),
		"in_flight":      rt.eng.InFlight(),
		"uptime_seconds": int64(time.Since(rt.started).Seconds()),
	})
}

// ListModels returns the served models.
func (rt *Router) ListModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, api.ModelList{
		Object: "list",
		Data: []api.Model{{
			ID:      rt.eng.ModelName(),
			Object:  "model",
			Created: rt.started.Unix(),
			OwnedBy: "fastmlx",
		}},
	})
}

// Embeddings is a stub until the embedding engine lands (spec milestone).
func (rt *Router) Embeddings(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		"embeddings are served by the embedding engine, which lands in a later milestone",
		"not_implemented_error", "")
}

// resolveSampling applies the sampling cascade: request param -> engine defaults
// -> hardcoded fallback. Only fields explicitly set on the request override the
// defaults (pointer fields distinguish unset from zero).
func (rt *Router) resolveSampling(maxTokens *int, maxCompletion *int, temp, topP, minP, repPen, presPen, freqPen *float64, topK *int, seed *int64, stop []string) engine.SamplingParams {
	sp := rt.eng.Defaults()
	if sp.Temperature == 0 && sp.TopP == 0 && sp.RepetitionPenalty == 0 {
		// No engine defaults configured: use the hardcoded fallback.
		sp = engine.SamplingParams{Temperature: 1.0, TopP: 0.95, TopK: 0, RepetitionPenalty: 1.0, MaxTokens: 32768}
	}
	if temp != nil {
		sp.Temperature = *temp
	}
	if topP != nil {
		sp.TopP = *topP
	}
	if topK != nil {
		sp.TopK = *topK
	}
	if minP != nil {
		sp.MinP = *minP
	}
	if repPen != nil {
		sp.RepetitionPenalty = *repPen
	}
	if presPen != nil {
		sp.PresencePenalty = *presPen
	}
	if freqPen != nil {
		sp.FrequencyPenalty = *freqPen
	}
	if seed != nil {
		sp.Seed = seed
	}
	if maxCompletion != nil {
		sp.MaxTokens = *maxCompletion
	} else if maxTokens != nil {
		sp.MaxTokens = *maxTokens
	}
	if sp.MaxTokens <= 0 {
		sp.MaxTokens = 32768
	}
	sp.Stop = stop
	return sp
}

// newID returns an OpenAI-style identifier with the given prefix.
func newID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg, typ, code string) {
	writeJSON(w, status, api.NewError(msg, typ, code))
}
