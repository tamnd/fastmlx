// SPDX-License-Identifier: MIT OR Apache-2.0

// Package scheduler implements continuous batching over a pipeline.DecodeStrategy.
// It runs the step loop (schedule waiting -> decode -> process
// responses -> cleanup) but depends only on the decode seam, so the mock backend
// and the real compute.BatchGenerator swap with zero scheduler changes.
package scheduler

import (
	"slices"
	"strings"
	"sync"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/tokenizer"
)

// SchedulingPolicy selects which waiting request runs next.
type SchedulingPolicy int

const (
	// PolicyFCFS serves requests first-come-first-served (the default).
	PolicyFCFS SchedulingPolicy = iota
)

// Config holds the runtime scheduling knobs. It is distinct from
// config.SchedulerConfig (the on-disk file schema): this is what the engine
// resolves and hands to the scheduler.
type Config struct {
	MaxNumSeqs          int // max concurrent sequences in the batch
	MaxNumBatchedTokens int // token budget per step (prefill accounting)
	CompletionBatchSize int
	EmbeddingBatchSize  int
	EnablePrefixCache   bool
	InitialCacheBlocks  int
	Policy              SchedulingPolicy
}

// DefaultConfig returns the scheduler defaults.
func DefaultConfig() Config {
	return Config{
		MaxNumSeqs:          8,
		MaxNumBatchedTokens: 8192,
		CompletionBatchSize: 8,
		EmbeddingBatchSize:  32,
		Policy:              PolicyFCFS,
	}
}

// Output is the result of one Step: the per-request increments produced this
// cycle, in a stable order.
type Output struct {
	Outputs []engine.RequestOutput
}

// seqState carries the per-request runtime that lives only while a request is
// being decoded: its backend UID and an incremental detokenizer.
type seqState struct {
	req   *engine.Request
	uid   int
	detok tokenizer.IncrementalDetokenizer
}

// Scheduler owns the waiting/running sets and drives one decode step at a time.
// All scheduling methods (AddRequest, Step) must run on a single goroutine - the
// engine core loop. Abort is the one exception and is goroutine-safe.
type Scheduler struct {
	cfg    Config
	decode pipeline.DecodeStrategy
	tok    tokenizer.Tokenizer

	waiting  []*engine.Request
	running  map[string]*seqState
	uidToReq map[int]string

	abortMu sync.Mutex
	aborts  map[string]struct{}
}

// New builds a scheduler over a decode backend and tokenizer.
func New(cfg Config, decode pipeline.DecodeStrategy, tok tokenizer.Tokenizer) *Scheduler {
	if cfg.MaxNumSeqs <= 0 {
		cfg.MaxNumSeqs = 1
	}
	return &Scheduler{
		cfg:      cfg,
		decode:   decode,
		tok:      tok,
		running:  make(map[string]*seqState),
		uidToReq: make(map[int]string),
		aborts:   make(map[string]struct{}),
	}
}

// AddRequest enqueues a request. It tokenizes the prompt if needed and records
// the prompt-token count. Called only on the engine core loop goroutine.
func (s *Scheduler) AddRequest(req *engine.Request) {
	if len(req.PromptTokenIDs) == 0 {
		switch p := req.Prompt.(type) {
		case string:
			req.PromptTokenIDs = s.tok.Encode(p)
		case []int:
			req.PromptTokenIDs = p
		}
	}
	req.NumPromptTokens = len(req.PromptTokenIDs)
	req.Status = engine.StatusWaiting
	s.waiting = append(s.waiting, req)
}

// Abort marks a request for cancellation. Goroutine-safe; the cancellation is
// applied on the next Step.
func (s *Scheduler) Abort(id string) {
	s.abortMu.Lock()
	s.aborts[id] = struct{}{}
	s.abortMu.Unlock()
}

// HasUnfinished reports whether the scheduler still has work to drive.
func (s *Scheduler) HasUnfinished() bool {
	return len(s.waiting) > 0 || len(s.running) > 0
}

// Step runs one scheduling cycle and returns the increments produced.
func (s *Scheduler) Step() (Output, error) {
	var out Output

	// 1. Apply pending aborts (waiting + running) before scheduling.
	out.Outputs = append(out.Outputs, s.applyAborts()...)

	// 2. Admit waiting requests up to MaxNumSeqs.
	s.scheduleWaiting()

	// 3. Decode one token per running sequence.
	if !s.decode.HasActive() {
		return out, nil
	}
	results, err := s.decode.Step()
	if err != nil {
		// Surface the backend error to every in-flight request as a 503 so the
		// loop never hangs even when the backend errors mid-step.
		out.Outputs = append(out.Outputs, s.failAll(err)...)
		return out, err
	}

	// 4. Turn decoded tokens into per-request increments.
	for _, r := range results {
		if o, ok := s.processToken(r); ok {
			out.Outputs = append(out.Outputs, o)
		}
	}
	return out, nil
}

// scheduleWaiting moves waiting requests into the running batch.
func (s *Scheduler) scheduleWaiting() {
	for len(s.waiting) > 0 && len(s.running) < s.cfg.MaxNumSeqs {
		req := s.waiting[0]
		s.waiting = s.waiting[1:]

		uid, err := s.decode.Insert(pipeline.DecodeRequest{
			Tokens:    req.PromptTokenIDs,
			MaxTokens: req.Sampling.MaxTokens,
		})
		if err != nil {
			req.Status = engine.StatusFinishedError
			// Re-queue handling is out of scope for the mock; drop with error.
			continue
		}
		req.Status = engine.StatusRunning
		req.BatchUID = uid
		st := &seqState{req: req, uid: uid, detok: s.tok.NewIncrementalDetokenizer()}
		s.running[req.ID] = st
		s.uidToReq[uid] = req.ID
	}
}

// processToken appends one decoded token to its request, detokenizes the delta,
// applies stop conditions, and returns the increment to emit.
func (s *Scheduler) processToken(r pipeline.TokenResult) (engine.RequestOutput, bool) {
	id, ok := s.uidToReq[r.UID]
	if !ok {
		return engine.RequestOutput{}, false
	}
	st := s.running[id]
	req := st.req

	finished := false
	reason := r.FinishReason

	// EOS / explicit stop token IDs end the sequence without emitting their text.
	isStopTok := r.Token == s.tok.EOSTokenID() || slices.Contains(req.Sampling.StopTokenIDs, r.Token)
	if isStopTok {
		finished = true
		if reason == "" {
			reason = "stop"
		}
	} else {
		req.OutputTokenIDs = append(req.OutputTokenIDs, r.Token)
	}

	delta := ""
	if !isStopTok {
		delta = st.detok.AddToken(r.Token)
		req.OutputText += delta
	}

	// Stop strings: truncate at the first match and finish (the stop text is not
	// included in the output, matching OpenAI semantics).
	if !finished {
		if cut, hit := firstStop(req.OutputText, req.Sampling.Stop); hit {
			// Trim the delta so we don't stream past the stop boundary.
			keep := max(len(cut)-(len(req.OutputText)-len(delta)), 0)
			if keep < len(delta) {
				delta = delta[:keep]
			}
			req.OutputText = cut
			finished = true
			reason = "stop"
		}
	}

	// Backend-declared finish (mock "length"/"stop") or MaxTokens enforcement.
	if !finished && r.FinishReason != "" {
		finished = true
	}
	if !finished && req.Sampling.MaxTokens > 0 && len(req.OutputTokenIDs) >= req.Sampling.MaxTokens {
		finished = true
		reason = "length"
	}

	o := engine.RequestOutput{
		RequestID:        req.ID,
		NewTokenIDs:      []int{r.Token},
		NewText:          delta,
		OutputTokenIDs:   req.OutputTokenIDs,
		OutputText:       req.OutputText,
		PromptTokens:     req.NumPromptTokens,
		CompletionTokens: len(req.OutputTokenIDs),
		CachedTokens:     req.CachedTokens,
	}
	if isStopTok {
		o.NewTokenIDs = nil
	}
	if finished {
		o.Finished = true
		o.FinishReason = reason
		req.FinishReason = reason
		req.Status = statusFor(reason)
		s.retire(req.ID, st.uid)
	}
	return o, true
}

// applyAborts cancels any requests marked for abort, in both the waiting and
// running sets, and returns the abort increments to emit.
func (s *Scheduler) applyAborts() []engine.RequestOutput {
	s.abortMu.Lock()
	if len(s.aborts) == 0 {
		s.abortMu.Unlock()
		return nil
	}
	pending := s.aborts
	s.aborts = make(map[string]struct{})
	s.abortMu.Unlock()

	var outs []engine.RequestOutput
	// Drop waiting requests.
	if len(s.waiting) > 0 {
		kept := s.waiting[:0]
		for _, req := range s.waiting {
			if _, ok := pending[req.ID]; ok {
				req.Status = engine.StatusFinishedAborted
				outs = append(outs, abortOutput(req))
			} else {
				kept = append(kept, req)
			}
		}
		s.waiting = kept
	}
	// Cancel running requests.
	for id := range pending {
		if st, ok := s.running[id]; ok {
			st.req.Status = engine.StatusFinishedAborted
			outs = append(outs, abortOutput(st.req))
			s.retire(id, st.uid)
		}
	}
	return outs
}

// failAll emits a 503-mapped error increment to every in-flight request.
func (s *Scheduler) failAll(err error) []engine.RequestOutput {
	var outs []engine.RequestOutput
	for id, st := range s.running {
		st.req.Status = engine.StatusFinishedError
		outs = append(outs, engine.RequestOutput{
			RequestID:    id,
			Finished:     true,
			FinishReason: "error",
			Err:          err.Error(),
		})
		s.retire(id, st.uid)
	}
	return outs
}

// retire removes a finished request from the running set and the backend.
func (s *Scheduler) retire(id string, uid int) {
	s.decode.Remove(uid)
	delete(s.running, id)
	delete(s.uidToReq, uid)
}

func abortOutput(req *engine.Request) engine.RequestOutput {
	return engine.RequestOutput{
		RequestID:        req.ID,
		OutputTokenIDs:   req.OutputTokenIDs,
		OutputText:       req.OutputText,
		Finished:         true,
		FinishReason:     "abort",
		PromptTokens:     req.NumPromptTokens,
		CompletionTokens: len(req.OutputTokenIDs),
	}
}

func statusFor(reason string) engine.RequestStatus {
	switch reason {
	case "length":
		return engine.StatusFinishedLength
	case "abort":
		return engine.StatusFinishedAborted
	case "error":
		return engine.StatusFinishedError
	default:
		return engine.StatusFinishedStopped
	}
}

// firstStop returns the output truncated at the earliest stop substring and
// whether any matched.
func firstStop(text string, stops []string) (string, bool) {
	best := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if i := strings.Index(text, stop); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	if best < 0 {
		return text, false
	}
	return text[:best], true
}
