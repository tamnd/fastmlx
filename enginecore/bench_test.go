// SPDX-License-Identifier: MIT OR Apache-2.0

package enginecore

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/scheduler"
	"github.com/tamnd/fastmlx/tokenizer"
)

// BenchmarkEngineConcurrentThroughput drives many requests through the engine
// concurrently and drains each stream to completion. This is the headline
// metric the rewrite targets: requests/sec under load, where Python's GIL
// serializes the scheduler and detokenizer.
func BenchmarkEngineConcurrentThroughput(b *testing.B) {
	for _, conc := range []int{1, 8, 32} {
		b.Run("conc="+strconv.Itoa(conc), func(b *testing.B) {
			tok := tokenizer.NewMock()
			cfg := scheduler.DefaultConfig()
			cfg.MaxNumSeqs = conc
			e := NewBatchedEngine(Options{
				ModelName: "mock",
				Tokenizer: tok,
				Decode:    pipeline.NewMockDecode(tok, "a short canned reply"),
				Scheduler: cfg,
			})
			ctx, cancel := context.WithCancel(context.Background())
			e.Start(ctx)
			defer func() { cancel(); e.Stop() }()

			b.ReportAllocs()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					i++
					id := strconv.Itoa(i) + "-" + strconv.FormatInt(time.Now().UnixNano(), 36)
					ch, err := e.Submit(&engine.Request{
						ID:       id,
						Prompt:   "p",
						Sampling: engine.SamplingParams{MaxTokens: 64},
					})
					if err != nil {
						// Admission rejection under load is expected; just retry next loop.
						continue
					}
					for range ch {
					}
				}
			})
		})
	}
}

// TestCoreBackpressureCoalescing exercises the slow-consumer path: a request
// whose channel fills must coalesce increments rather than block the loop, and
// the consumer must still see the complete concatenated text once it drains.
func TestCoreBackpressureCoalescing(t *testing.T) {
	tok := tokenizer.NewMock()
	// A long response forces many increments past the 64-slot channel buffer.
	long := strings.Repeat("x", 500)
	e := NewBatchedEngine(Options{
		ModelName: "mock",
		Tokenizer: tok,
		Decode:    pipeline.NewMockDecode(tok, long),
	})
	ctx, cancel := context.WithCancel(context.Background())
	e.Start(ctx)
	defer func() { cancel(); e.Stop() }()

	ch, err := e.Submit(&engine.Request{ID: "slow", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 1000}})
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately read slowly so the channel backs up and the loop coalesces.
	var text string
	var last engine.RequestOutput
	first := true
	for o := range ch {
		text += o.NewText
		last = o
		if first {
			time.Sleep(20 * time.Millisecond)
			first = false
		}
	}
	if !last.Finished {
		t.Fatal("stream ended without a finished increment")
	}
	if len(text) != 500 {
		t.Errorf("coalesced text length = %d, want 500 (no tokens dropped)", len(text))
	}
}

func TestCoreSubmitAfterStop(t *testing.T) {
	tok := tokenizer.NewMock()
	e := NewBatchedEngine(Options{ModelName: "mock", Tokenizer: tok, Decode: pipeline.NewMockDecode(tok, "hi")})
	ctx, cancel := context.WithCancel(context.Background())
	e.Start(ctx)
	cancel()
	e.Stop()
	_, err := e.Submit(&engine.Request{ID: "late", Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 10}})
	if err != ErrNotRunning {
		t.Fatalf("expected ErrNotRunning after Stop, got %v", err)
	}
}

func TestCoreInFlightReturnsToZero(t *testing.T) {
	tok := tokenizer.NewMock()
	e := NewBatchedEngine(Options{ModelName: "mock", Tokenizer: tok, Decode: pipeline.NewMockDecode(tok, "done")})
	ctx, cancel := context.WithCancel(context.Background())
	e.Start(ctx)
	defer func() { cancel(); e.Stop() }()

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ch, err := e.Submit(&engine.Request{ID: "r" + strconv.Itoa(i), Prompt: "p", Sampling: engine.SamplingParams{MaxTokens: 20}})
			if err != nil {
				return
			}
			for range ch {
			}
		}(i)
	}
	wg.Wait()

	// After every stream drains, the in-flight counter must settle to zero.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if e.InFlight() == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("InFlight = %d after draining all requests, want 0", e.InFlight())
}
