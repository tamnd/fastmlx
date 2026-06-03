// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"errors"
	"sort"
	"sync"

	"github.com/tamnd/fastmlx/mlxgo"
	"github.com/tamnd/fastmlx/pipeline"
)

// Model is the per-step forward the BatchGenerator drives. A concrete model from
// compute/models satisfies it through a thin adapter. The cache and the logits
// are opaque (any) so the seam stays GPU-native: under -tags mlx Forward returns
// an mlx_array logits row and the Sampler runs a categorical kernel on it, while
// on the default stub Forward returns ErrMLXUnavailable at the first kernel op.
// Nothing about the continuous-batching bookkeeping needs to read the tensor, so
// the generator never forces a host readback.
type Model interface {
	// NewCache allocates a fresh per-sequence KV cache set for one sequence.
	NewCache() any
	// Forward runs one sequence's pending tokens through the model with its
	// cache and returns the logits row for the final position. The cache is the
	// value NewCache produced for this sequence; Forward appends the step's keys
	// and values to it. tokens is the slice fed this step (the whole prompt on
	// the prefill step, then one token per decode step).
	Forward(tokens []int32, cache any, s *mlxgo.Stream) (logits any, err error)
	// EOS reports the end-of-sequence token id that finishes a sequence.
	EOS() int
}

// ErrEmptyPrompt is returned by Insert when a request carries no prompt tokens.
// A forward pass needs at least one token to produce its first logits row.
var ErrEmptyPrompt = errors.New("compute: decode request has no prompt tokens")

// BatchGenerator is the continuous-batching token generator: a set of in-flight
// sequences, each with its own KV cache and sampler, advanced one token per Step.
// It is the single seam the scheduler depends on (pipeline.DecodeStrategy), so the
// stage-2 MockDecode and this stage-4 backend are interchangeable with zero
// scheduler changes.
//
// The bookkeeping here is GPU-free and host-testable: which sequences are active,
// each one's token history, how many tokens have been fed to the model (the cache
// offset), how many have been generated, and the EOS / max-length finish
// detection. The only GPU-bound work is Model.Forward and the Sampler, both behind
// the any-typed seam; on the default stub a Step returns ErrMLXUnavailable from
// the first Forward after running all of the host-side gather and lifecycle
// bookkeeping.
type BatchGenerator struct {
	model Model
	s     *mlxgo.Stream

	mu      sync.Mutex
	nextUID int
	active  map[int]*bgSeq
}

// bgSeq is one in-flight sequence's state. pending holds the tokens sampled but
// not yet fed to the model: the whole prompt before the first Step, then the
// single token sampled by the previous Step. offset is the number of tokens
// already fed (the cache length), which the model reads as the RoPE position.
type bgSeq struct {
	pending   []int32
	generated int
	maxTokens int
	offset    int
	sampler   pipeline.Sampler
	procs     []pipeline.LogitsProc
	cache     any
	finished  bool
}

// NewBatchGenerator builds a generator driving model. The stream is created once
// and reused for every forward; on the stub it is an inert handle.
func NewBatchGenerator(model Model) (*BatchGenerator, error) {
	s, err := mlxgo.NewStream()
	if err != nil {
		return nil, err
	}
	return &BatchGenerator{
		model:  model,
		s:      s,
		active: make(map[int]*bgSeq),
	}, nil
}

// Insert admits a sequence and returns its backend UID. It allocates the
// per-sequence KV cache (reusing req.Cache from a prefix-cache hit when present)
// and queues the whole prompt as the prefill batch.
func (g *BatchGenerator) Insert(req pipeline.DecodeRequest) (int, error) {
	if len(req.Tokens) == 0 {
		return 0, ErrEmptyPrompt
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	cache := req.Cache
	if cache == nil {
		cache = g.model.NewCache()
	}
	pending := make([]int32, len(req.Tokens))
	for i, t := range req.Tokens {
		pending[i] = int32(t)
	}
	uid := g.nextUID
	g.nextUID++
	g.active[uid] = &bgSeq{
		pending:   pending,
		maxTokens: req.MaxTokens,
		sampler:   req.Sampler,
		procs:     req.LogitsProcs,
		cache:     cache,
	}
	return uid, nil
}

// Step advances every active, unfinished sequence by one token. UIDs are visited
// in a stable order so output is deterministic and the per-row gather matches the
// batched layout the GPU backend builds. The first error from any sequence's
// forward aborts the step (the stub returns ErrMLXUnavailable here).
func (g *BatchGenerator) Step() ([]pipeline.TokenResult, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(g.active) == 0 {
		return nil, nil
	}
	uids := make([]int, 0, len(g.active))
	for uid := range g.active {
		uids = append(uids, uid)
	}
	sort.Ints(uids)

	results := make([]pipeline.TokenResult, 0, len(uids))
	for _, uid := range uids {
		seq := g.active[uid]
		if seq.finished {
			continue
		}
		logits, err := g.model.Forward(seq.pending, seq.cache, g.s)
		if err != nil {
			return nil, err
		}
		for _, p := range seq.procs {
			logits = p.Apply(logits)
		}
		tok := seq.sampler.Sample(logits)

		seq.offset += len(seq.pending)
		seq.pending = []int32{int32(tok)}
		seq.generated++

		res := pipeline.TokenResult{UID: uid, Token: tok}
		switch {
		case tok == g.model.EOS():
			res.FinishReason = "stop"
		case seq.maxTokens > 0 && seq.generated >= seq.maxTokens:
			res.FinishReason = "length"
		}
		if res.FinishReason != "" {
			seq.finished = true
			res.PromptCache = seq.cache
		}
		results = append(results, res)
	}
	return results, nil
}

// Remove retires a sequence and returns its KV cache for prefix-cache storage.
// It is safe to call for an unknown UID (returns nil).
func (g *BatchGenerator) Remove(uid int) any {
	g.mu.Lock()
	defer g.mu.Unlock()
	seq, ok := g.active[uid]
	if !ok {
		return nil
	}
	delete(g.active, uid)
	return seq.cache
}

// HasActive reports whether any sequence is still decoding (a finished sequence
// awaiting Remove still counts as active).
func (g *BatchGenerator) HasActive() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.active) > 0
}

// Close drops all in-flight sequences and releases their caches.
func (g *BatchGenerator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.active = make(map[int]*bgSeq)
	return nil
}

// compile-time check: the generator is a drop-in DecodeStrategy.
var _ pipeline.DecodeStrategy = (*BatchGenerator)(nil)
