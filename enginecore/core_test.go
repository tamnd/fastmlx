// SPDX-License-Identifier: MIT OR Apache-2.0

package enginecore

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/scheduler"
	"github.com/tamnd/fastmlx/tokenizer"
)

func newEngine(t *testing.T, resp string, maxConc int) (*BatchedEngine, context.CancelFunc) {
	t.Helper()
	tok := tokenizer.NewMock()
	e := NewBatchedEngine(Options{
		ModelName:     "mock",
		Tokenizer:     tok,
		Decode:        pipeline.NewMockDecode(tok, resp),
		Scheduler:     scheduler.DefaultConfig(),
		MaxConcurrent: maxConc,
	})
	ctx, cancel := context.WithCancel(context.Background())
	e.Start(ctx)
	return e, cancel
}

func collect(ch <-chan engine.RequestOutput) engine.RequestOutput {
	var last engine.RequestOutput
	var text string
	for o := range ch {
		text += o.NewText
		last = o
	}
	last.OutputText = text
	return last
}

func TestCoreSingleRequest(t *testing.T) {
	e, cancel := newEngine(t, "hello there", 8)
	defer cancel()

	ch, err := e.Submit(&engine.Request{ID: "r1", Prompt: "hi", Sampling: engine.SamplingParams{MaxTokens: 100}})
	if err != nil {
		t.Fatal(err)
	}
	out := collect(ch)
	if !out.Finished || out.FinishReason != "stop" {
		t.Fatalf("finished=%v reason=%q", out.Finished, out.FinishReason)
	}
	if out.OutputText != "hello there" {
		t.Errorf("streamed text = %q, want %q", out.OutputText, "hello there")
	}
}

func TestCoreConcurrentRequests(t *testing.T) {
	e, cancel := newEngine(t, "batched output", 32)
	defer cancel()

	var wg sync.WaitGroup
	const n = 16
	results := make([]string, n)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := "r" + string(rune('A'+i))
			ch, err := e.Submit(&engine.Request{ID: id, Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 100}})
			if err != nil {
				t.Errorf("submit %s: %v", id, err)
				return
			}
			results[i] = collect(ch).OutputText
		}(i)
	}
	wg.Wait()
	for i, got := range results {
		if got != "batched output" {
			t.Errorf("request %d output = %q", i, got)
		}
	}
}

func TestCoreAdmissionControl(t *testing.T) {
	// One concurrent slot; a long response keeps it busy.
	e, cancel := newEngine(t, "a response long enough to occupy the slot for a while", 1)
	defer cancel()

	ch1, err := e.Submit(&engine.Request{ID: "r1", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})
	if err != nil {
		t.Fatal(err)
	}
	// Second submit should be rejected while the slot is occupied.
	_, err = e.Submit(&engine.Request{ID: "r2", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})
	if err != ErrQueueFull {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
	collect(ch1) // drain r1 so the slot frees

	// Now a new request is admitted.
	ch3, err := e.Submit(&engine.Request{ID: "r3", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 5}})
	if err != nil {
		t.Fatalf("expected admission after slot freed, got %v", err)
	}
	collect(ch3)
}

func TestCoreAbort(t *testing.T) {
	e, cancel := newEngine(t, "long response repeated so abort lands mid-stream xxxxxxxxxx", 8)
	defer cancel()

	ch, err := e.Submit(&engine.Request{ID: "r1", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 100000}})
	if err != nil {
		t.Fatal(err)
	}
	// Read one increment, then abort.
	<-ch
	e.Abort("r1")
	out := collect(ch)
	if out.FinishReason != "abort" && out.FinishReason != "stop" {
		// stop is acceptable if generation finished before the abort landed.
		t.Fatalf("reason = %q, want abort", out.FinishReason)
	}
}

func TestCoreStop(t *testing.T) {
	tok := tokenizer.NewMock()
	e := NewBatchedEngine(Options{ModelName: "mock", Tokenizer: tok, Decode: pipeline.NewMockDecode(tok, "x")})
	ctx, cancel := context.WithCancel(context.Background())
	e.Start(ctx)
	cancel()
	done := make(chan struct{})
	go func() { e.Stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Stop did not return after context cancel")
	}
}
