// SPDX-License-Identifier: Apache-2.0

// Package engine defines the request lifecycle types and the Engine interface
// shared by every model backend (batched LLM, VLM, embedding, reranker, audio).
// These types describe the request lifecycle shared by every backend.
package engine

import "time"

// RequestStatus is the lifecycle state of a request.
type RequestStatus int

const (
	StatusWaiting RequestStatus = iota
	StatusPrefilling
	StatusRunning
	StatusPreempted
	StatusFinishedStopped
	StatusFinishedLength
	StatusFinishedAborted
	StatusFinishedError
)

func (s RequestStatus) Finished() bool { return s >= StatusFinishedStopped }

// SamplingParams controls token sampling. Penalty/temperature defaults follow
// convention (multiplicative repetition penalty, 1.0 = off; OpenAI-additive presence/
// frequency). A nil Seed means unseeded.
type SamplingParams struct {
	MaxTokens         int
	Temperature       float64
	TopP              float64
	TopK              int
	MinP              float64
	RepetitionPenalty float64
	PresencePenalty   float64
	FrequencyPenalty  float64
	Seed              *int64
	Stop              []string
	StopTokenIDs      []int
	GuidedGrammar     string
}

// Request is a single in-flight generation request.
type Request struct {
	ID              string
	Prompt          any // string | []int
	PromptTokenIDs  []int
	NumPromptTokens int
	Sampling        SamplingParams
	Arrival         time.Time
	Status          RequestStatus
	NumComputed     int
	OutputTokenIDs  []int
	OutputText      string
	BatchUID        int
	PromptCache     any
	CachedTokens    int
	FirstTokenTime  time.Time
	FinishReason    string
}

// RequestOutput is one increment (or the final result) for a request, delivered
// over the engine's per-request output channel.
// CachedTokens / CacheCreation feed the OpenAI usage cache fields.
type RequestOutput struct {
	RequestID        string
	NewTokenIDs      []int
	NewText          string
	OutputTokenIDs   []int
	OutputText       string
	Finished         bool
	FinishReason     string // "stop" | "length" | "abort" | "error"
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	CacheCreation    int
	Err              string // non-empty -> engine abort, maps to HTTP 503 not 500
}
