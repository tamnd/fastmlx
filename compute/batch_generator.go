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

// BatchDecoder is the optional capability a Model implements to decode a
// synchronized batch of sequences in a single forward. When the generator finds
// every in-flight sequence on the same decode step (one pending token and an
// identical cache offset), it calls BatchDecode once instead of Forward per
// sequence, which is the throughput path the 2x concurrency goal rests on. The
// per-sequence path stays the fallback for prefill steps and for a backend that
// does not implement this.
//
// tokens[i] and caches[i] belong to sequence i in the same order; caches[i] is the
// value NewCache produced for that sequence, which BatchDecode merges along the
// batch axis, advances by one step, and writes back. The result holds one logits
// row per sequence, again in order, each the opaque value a Sampler consumes.
type BatchDecoder interface {
	BatchDecode(tokens []int32, caches []any, s *mlxgo.Stream) (logits []any, err error)
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

	live := make([]int, 0, len(uids))
	for _, uid := range uids {
		if !g.active[uid].finished {
			live = append(live, uid)
		}
	}

	if batched, results, err := g.tryBatchedStep(live); batched {
		return results, err
	}

	results := make([]pipeline.TokenResult, 0, len(live))
	for _, uid := range live {
		seq := g.active[uid]
		advance := len(seq.pending)
		logits, err := g.model.Forward(seq.pending, seq.cache, g.s)
		if err != nil {
			return nil, err
		}
		results = append(results, g.consume(uid, seq, logits, advance))
	}
	return results, nil
}

// tryBatchedStep decodes the live cohort in one forward when it is fully
// synchronized: at least two sequences, every one on a single pending decode
// token at the same cache offset, and the model implementing BatchDecoder. That
// is exactly the state a batch reaches once every prompt has been prefilled and
// the rows step in lockstep, which is the regime the throughput goal targets. It
// returns batched=false to fall back to the per-sequence path on any prefill
// step, a ragged cohort (sequences at different offsets, the heterogeneous-length
// case a right-pad-prefill admission redesign will fold in later), a lone
// sequence, or a backend without the capability. When it returns batched=true the
// step is fully handled, error included.
func (g *BatchGenerator) tryBatchedStep(live []int) (bool, []pipeline.TokenResult, error) {
	bd, ok := g.model.(BatchDecoder)
	if !ok || len(live) < 2 {
		return false, nil, nil
	}
	offset := g.active[live[0]].offset
	for _, uid := range live {
		seq := g.active[uid]
		if len(seq.pending) != 1 || seq.offset != offset {
			return false, nil, nil
		}
	}

	tokens := make([]int32, len(live))
	caches := make([]any, len(live))
	for i, uid := range live {
		seq := g.active[uid]
		tokens[i] = seq.pending[0]
		caches[i] = seq.cache
	}
	rows, err := bd.BatchDecode(tokens, caches, g.s)
	if err != nil {
		return true, nil, err
	}

	results := make([]pipeline.TokenResult, len(live))
	for i, uid := range live {
		seq := g.active[uid]
		results[i] = g.consume(uid, seq, rows[i], 1)
	}
	return true, results, nil
}

// consume turns one sequence's fresh logits into the next token and advances the
// sequence: it runs the logits processors, samples, moves the cache offset on by
// the tokens just fed, queues the sampled token as the next pending input, and
// detects an EOS or max-length finish. It is the shared tail of both the
// per-sequence Forward path (advance is the prompt or single-token length) and the
// batched path (advance is always one), so the two paths stay byte-for-byte
// identical past the forward.
func (g *BatchGenerator) consume(uid int, seq *bgSeq, logits any, advance int) pipeline.TokenResult {
	for _, p := range seq.procs {
		logits = p.Apply(logits)
	}
	tok := seq.sampler.Sample(logits)

	seq.offset += advance
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
	return res
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
