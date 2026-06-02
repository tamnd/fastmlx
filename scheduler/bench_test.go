// SPDX-License-Identifier: MIT OR Apache-2.0

package scheduler

import (
	"strconv"
	"testing"

	"github.com/tamnd/fastmlx/engine"
)

// runBatch drives a fresh batch of n requests to completion and reports the
// total increments emitted, so the benchmark measures the full schedule->decode
// ->process->retire cycle and not just a single Step.
func runBatch(s *Scheduler, n int) int {
	for i := range n {
		s.AddRequest(&engine.Request{
			ID:       "r" + strconv.Itoa(i),
			Prompt:   "prompt",
			Sampling: engine.SamplingParams{MaxTokens: 64},
		})
	}
	total := 0
	for s.HasUnfinished() {
		out, err := s.Step()
		if err != nil {
			break
		}
		total += len(out.Outputs)
	}
	return total
}

// BenchmarkSchedulerThroughput measures end-to-end scheduling cost at the
// default batch width. Each iteration runs a full batch to completion.
func BenchmarkSchedulerThroughput(b *testing.B) {
	for _, width := range []int{1, 8, 32} {
		b.Run("seqs="+strconv.Itoa(width), func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.MaxNumSeqs = width
			b.ReportAllocs()
			for b.Loop() {
				s, _ := newSched("the quick brown fox jumps over the lazy dog", cfg)
				runBatch(s, width)
			}
		})
	}
}

// BenchmarkSchedulerStep isolates a single Step over a steady-state batch with
// no admission or retirement churn, approximating the decode-loop hot path.
func BenchmarkSchedulerStep(b *testing.B) {
	cfg := DefaultConfig()
	cfg.MaxNumSeqs = 16
	s, _ := newSched("a fairly long canned response to keep every sequence busy across many steps", cfg)
	for i := range 16 {
		s.AddRequest(&engine.Request{
			ID:       "r" + strconv.Itoa(i),
			Prompt:   "p",
			Sampling: engine.SamplingParams{MaxTokens: 1 << 20},
		})
	}
	// Prime the batch so all sequences are running before timing.
	s.applyAborts()
	s.scheduleWaiting()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := s.Step(); err != nil {
			b.Fatal(err)
		}
	}
}

func TestSchedulerStopTokenID(t *testing.T) {
	// The mock emits rune code points; pick the code point of 'z' as a stop token
	// and a response containing it so the sequence ends on that token (text excluded).
	s, _ := newSched("abz", DefaultConfig())
	s.AddRequest(&engine.Request{
		ID:       "r1",
		Prompt:   "p",
		Sampling: engine.SamplingParams{MaxTokens: 100, StopTokenIDs: []int{int('z')}},
	})
	o := drain(s)["r1"]
	if o.FinishReason != "stop" {
		t.Fatalf("reason = %q, want stop", o.FinishReason)
	}
	if o.OutputText != "ab" {
		t.Errorf("output = %q, want %q (stop token text excluded)", o.OutputText, "ab")
	}
}

func TestSchedulerMultipleStopStringsEarliestWins(t *testing.T) {
	s, _ := newSched("one END two HALT three", DefaultConfig())
	s.AddRequest(&engine.Request{
		ID:       "r1",
		Prompt:   "p",
		Sampling: engine.SamplingParams{MaxTokens: 100, Stop: []string{"HALT", "END"}},
	})
	o := drain(s)["r1"]
	if o.OutputText != "one " {
		t.Errorf("output = %q, want %q (earliest stop wins)", o.OutputText, "one ")
	}
}

func TestSchedulerEmptyPromptStillRuns(t *testing.T) {
	s, _ := newSched("hi", DefaultConfig())
	s.AddRequest(&engine.Request{ID: "r1", Prompt: "", Sampling: engine.SamplingParams{MaxTokens: 100}})
	o := drain(s)["r1"]
	if o.PromptTokens != 0 {
		t.Errorf("prompt tokens = %d, want 0", o.PromptTokens)
	}
	if o.OutputText != "hi" {
		t.Errorf("output = %q, want %q", o.OutputText, "hi")
	}
}

func TestSchedulerPreTokenizedPrompt(t *testing.T) {
	s, _ := newSched("hi", DefaultConfig())
	ids := []int{int('a'), int('b'), int('c')}
	s.AddRequest(&engine.Request{ID: "r1", Prompt: ids, Sampling: engine.SamplingParams{MaxTokens: 100}})
	o := drain(s)["r1"]
	if o.PromptTokens != 3 {
		t.Errorf("prompt tokens = %d, want 3 (pre-tokenized []int prompt)", o.PromptTokens)
	}
}

func TestSchedulerAbortWaitingRequest(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxNumSeqs = 1
	s, _ := newSched("long enough response to keep the single slot busy for a while", cfg)
	s.AddRequest(&engine.Request{ID: "running", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})
	s.AddRequest(&engine.Request{ID: "waiting", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})

	// Admit only "running" (cap 1); "waiting" stays queued.
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
	s.Abort("waiting")
	out, _ := s.Step()
	var abortedWaiting bool
	for _, o := range out.Outputs {
		if o.RequestID == "waiting" && o.FinishReason == "abort" {
			abortedWaiting = true
		}
	}
	if !abortedWaiting {
		t.Fatal("expected the queued request to abort before admission")
	}
}
