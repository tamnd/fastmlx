// SPDX-License-Identifier: Apache-2.0

// Package pipeline defines the decode seam between the scheduler and a compute
// backend. The scheduler depends only on DecodeStrategy, so the mock backend
// (this stage) and the real compute.BatchGenerator (the cgo/mlx-c backend) swap
// with zero scheduler changes. The "adding a model = zero scheduler changes"
// principle. It exposes an insert/step/remove surface.
package pipeline

// Sampler turns a backend's per-step logits into a chosen token. The mock backend
// ignores it; the real backend implements temperature/top-p/top-k/min-p. Kept as
// an open interface so the sampling cascade can evolve without touching the seam.
type Sampler interface {
	// Sample picks a token ID from backend-specific logits.
	Sample(logits any) int
}

// LogitsProc mutates logits before sampling (repetition penalty, grammar masks,
// tool-logits constraints). The mock backend ignores them.
type LogitsProc interface {
	Apply(logits any) any
}

// DecodeRequest is everything the backend needs to start decoding one sequence.
type DecodeRequest struct {
	Tokens      []int // prompt token IDs
	MaxTokens   int   // generation cap
	Sampler     Sampler
	LogitsProcs []LogitsProc
	Cache       any // KV state from a prefix-cache hit, or nil for a fresh sequence
}

// TokenResult is one decoded token for one active sequence in a step.
type TokenResult struct {
	UID          int
	Token        int
	Logprobs     any
	FinishReason string // "" while generating; "stop" on EOS, "length" at MaxTokens
	PromptCache  any    // KV state handed back when the sequence finishes
}

// DecodeStrategy is the batched-decode contract. One step advances every active
// sequence by one token (continuous batching).
type DecodeStrategy interface {
	// Insert admits a sequence and returns its backend UID.
	Insert(req DecodeRequest) (uid int, err error)
	// Step advances all active sequences by one token.
	Step() ([]TokenResult, error)
	// Remove retires a sequence and returns any KV cache to reclaim.
	Remove(uid int) (cache any)
	// HasActive reports whether any sequence is still decoding.
	HasActive() bool
	// Close releases backend resources.
	Close() error
}

// DecodePlugin wraps a DecodeStrategy step to implement speculative decoding
// (MTP, DFlash, SpecPrefill). Lands with the speculative milestone; defined here
// so the seam is stable.
type DecodePlugin interface {
	WrapStep(base func() ([]TokenResult, error)) ([]TokenResult, error)
	OnInsert(req DecodeRequest, uid int)
	OnRemove(uid int)
	OnClose()
}
