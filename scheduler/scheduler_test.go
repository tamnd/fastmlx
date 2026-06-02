// SPDX-License-Identifier: MIT OR Apache-2.0

package scheduler

import (
	"testing"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/tokenizer"
)

func drain(s *Scheduler) map[string]engine.RequestOutput {
	final := map[string]engine.RequestOutput{}
	for s.HasUnfinished() {
		out, err := s.Step()
		if err != nil {
			break
		}
		for _, o := range out.Outputs {
			if o.Finished {
				final[o.RequestID] = o
			}
		}
	}
	return final
}

func newSched(resp string, cfg Config) (*Scheduler, tokenizer.Tokenizer) {
	tok := tokenizer.NewMock()
	d := pipeline.NewMockDecode(tok, resp)
	return New(cfg, d, tok), tok
}

func TestSchedulerSingleRequest(t *testing.T) {
	s, tok := newSched("hello world", DefaultConfig())
	s.AddRequest(&engine.Request{ID: "r1", Prompt: "ignored", Sampling: engine.SamplingParams{MaxTokens: 100}})

	final := drain(s)
	o := final["r1"]
	if !o.Finished || o.FinishReason != "stop" {
		t.Fatalf("got finished=%v reason=%q", o.Finished, o.FinishReason)
	}
	if o.OutputText != "hello world" {
		t.Errorf("output = %q, want %q", o.OutputText, "hello world")
	}
	if o.CompletionTokens != len(tok.Encode("hello world")) {
		t.Errorf("completion tokens = %d", o.CompletionTokens)
	}
	if o.PromptTokens != len(tok.Encode("ignored")) {
		t.Errorf("prompt tokens = %d", o.PromptTokens)
	}
}

func TestSchedulerStopString(t *testing.T) {
	s, _ := newSched("alpha STOP beta", DefaultConfig())
	s.AddRequest(&engine.Request{
		ID:       "r1",
		Prompt:   "x",
		Sampling: engine.SamplingParams{MaxTokens: 100, Stop: []string{"STOP"}},
	})
	o := drain(s)["r1"]
	if o.FinishReason != "stop" {
		t.Fatalf("reason = %q, want stop", o.FinishReason)
	}
	if o.OutputText != "alpha " {
		t.Errorf("output = %q, want %q (stop text excluded)", o.OutputText, "alpha ")
	}
}

func TestSchedulerMaxTokensLength(t *testing.T) {
	s, _ := newSched("aaaaaaaaaa", DefaultConfig())
	s.AddRequest(&engine.Request{ID: "r1", Prompt: "x", Sampling: engine.SamplingParams{MaxTokens: 4}})
	o := drain(s)["r1"]
	if o.FinishReason != "length" {
		t.Fatalf("reason = %q, want length", o.FinishReason)
	}
	if o.CompletionTokens != 4 {
		t.Errorf("completion tokens = %d, want 4", o.CompletionTokens)
	}
}

func TestSchedulerConcurrentBatch(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxNumSeqs = 4
	s, _ := newSched("hi", cfg)
	for _, id := range []string{"a", "b", "c"} {
		s.AddRequest(&engine.Request{ID: id, Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 50}})
	}
	final := drain(s)
	if len(final) != 3 {
		t.Fatalf("finished %d requests, want 3", len(final))
	}
	for _, id := range []string{"a", "b", "c"} {
		if final[id].OutputText != "hi" {
			t.Errorf("req %s output = %q", id, final[id].OutputText)
		}
	}
}

func TestSchedulerAdmissionCap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxNumSeqs = 1
	s, _ := newSched("hello", cfg)
	s.AddRequest(&engine.Request{ID: "a", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 50}})
	s.AddRequest(&engine.Request{ID: "b", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 50}})

	// First step admits only one sequence.
	s.applyAborts()
	s.scheduleWaiting()
	if len(s.running) != 1 {
		t.Fatalf("running = %d, want 1 (admission cap)", len(s.running))
	}
	final := drain(s)
	if len(final) != 2 {
		t.Errorf("eventually finished %d, want 2", len(final))
	}
}

func TestSchedulerAbort(t *testing.T) {
	s, _ := newSched("this is a long enough response to span steps", DefaultConfig())
	s.AddRequest(&engine.Request{ID: "r1", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})

	// One step to admit + start decoding, then abort.
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
	s.Abort("r1")
	out, _ := s.Step()
	var aborted bool
	for _, o := range out.Outputs {
		if o.RequestID == "r1" && o.Finished && o.FinishReason == "abort" {
			aborted = true
		}
	}
	if !aborted {
		t.Fatal("expected abort increment")
	}
	if s.HasUnfinished() {
		t.Error("scheduler still has work after abort")
	}
}
