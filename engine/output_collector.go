// SPDX-License-Identifier: MIT OR Apache-2.0

// Adapted from the vllm-mlx RequestOutputCollector pattern.

package engine

import (
	"context"
	"sync"
	"sync/atomic"
)

// waitingConsumers counts collectors with a consumer blocked in Get across the
// whole process, mirroring the reference's class-level counter. The engine reads
// it to decide whether yielding is worthwhile.
var waitingConsumers int64

// RequestOutputCollector is a per-request, single-producer single-consumer
// output buffer for streaming. The producer (engine loop) calls Put; the
// consumer (streaming handler) drains with GetNowait and falls back to the
// blocking Get. When aggregate is set, a Put that arrives while an output is
// still buffered merges into it so a fast producer cannot explode the buffer.
//
// The reference signals readiness with an asyncio.Event; here a capacity-one
// channel plays that role so Get can also honour context cancellation.
type RequestOutputCollector struct {
	mu        sync.Mutex
	output    RequestOutput
	hasOutput bool
	aggregate bool
	isWaiting bool
	ready     chan struct{}
}

// NewRequestOutputCollector returns a collector. aggregate merges queued outputs
// when the producer gets ahead of the consumer.
func NewRequestOutputCollector(aggregate bool) *RequestOutputCollector {
	return &RequestOutputCollector{aggregate: aggregate, ready: make(chan struct{}, 1)}
}

// Put stores an output without blocking. With aggregation on and an output
// already buffered, the new output merges into it; otherwise it replaces.
func (c *RequestOutputCollector) Put(out RequestOutput) {
	c.mu.Lock()
	switch {
	case !c.hasOutput:
		c.output = out
		c.hasOutput = true
	case c.aggregate:
		c.output = mergeOutputs(c.output, out)
	default:
		c.output = out
	}
	c.signalReadyLocked()
	c.mu.Unlock()
}

// GetNowait returns the buffered output if present, clearing the buffer. The
// bool is false when nothing is buffered, avoiding a task switch under load.
func (c *RequestOutputCollector) GetNowait() (RequestOutput, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.takeLocked()
}

// Get returns the next output, blocking until one is available or ctx is done.
// For low latency the caller should try GetNowait first.
func (c *RequestOutputCollector) Get(ctx context.Context) (RequestOutput, error) {
	c.mu.Lock()
	if !c.isWaiting {
		c.isWaiting = true
		atomic.AddInt64(&waitingConsumers, 1)
	}
	c.mu.Unlock()

	defer c.stopWaiting()

	for {
		c.mu.Lock()
		if out, ok := c.takeLocked(); ok {
			c.mu.Unlock()
			return out, nil
		}
		c.mu.Unlock()

		select {
		case <-c.ready:
		case <-ctx.Done():
			return RequestOutput{}, ctx.Err()
		}
	}
}

// Clear drops any pending output and releases a waiting consumer count.
func (c *RequestOutputCollector) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.output = RequestOutput{}
	c.hasOutput = false
	c.drainReadyLocked()
	if c.isWaiting {
		c.isWaiting = false
		atomic.AddInt64(&waitingConsumers, -1)
	}
}

// HasWaitingConsumers reports whether any collector has a blocked consumer.
func HasWaitingConsumers() bool {
	return atomic.LoadInt64(&waitingConsumers) > 0
}

// takeLocked removes and returns the buffered output. The caller holds the lock.
func (c *RequestOutputCollector) takeLocked() (RequestOutput, bool) {
	if !c.hasOutput {
		return RequestOutput{}, false
	}
	out := c.output
	c.output = RequestOutput{}
	c.hasOutput = false
	c.drainReadyLocked()
	return out, true
}

// signalReadyLocked sets the readiness signal (idempotent, like Event.set).
func (c *RequestOutputCollector) signalReadyLocked() {
	select {
	case c.ready <- struct{}{}:
	default:
	}
}

// drainReadyLocked clears the readiness signal (like Event.clear).
func (c *RequestOutputCollector) drainReadyLocked() {
	select {
	case <-c.ready:
	default:
	}
}

func (c *RequestOutputCollector) stopWaiting() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.isWaiting {
		c.isWaiting = false
		atomic.AddInt64(&waitingConsumers, -1)
	}
}

// mergeOutputs combines two outputs when the producer gets ahead of the
// consumer: the new tokens and text concatenate while the cumulative fields and
// status take the latest values. The error is the new one when set, else the
// existing one is kept.
func mergeOutputs(existing, incoming RequestOutput) RequestOutput {
	merged := incoming
	merged.NewTokenIDs = append(append([]int{}, existing.NewTokenIDs...), incoming.NewTokenIDs...)
	merged.NewText = existing.NewText + incoming.NewText
	if incoming.Err == "" {
		merged.Err = existing.Err
	}
	return merged
}

// RequestStreamState batches streamed tokens by a stream interval, holding back
// output until enough tokens accumulate while always flushing the first token
// and the final one.
type RequestStreamState struct {
	StreamInterval int
	SentTokens     int
}

// NewRequestStreamState returns a stream state with the given interval. The
// reference defaults the interval to 1.
func NewRequestStreamState(streamInterval int) RequestStreamState {
	return RequestStreamState{StreamInterval: streamInterval}
}

// ShouldSend reports whether the accumulated output should be flushed: always on
// finish, always for the first token (low TTFT), otherwise once at least
// StreamInterval new tokens have accumulated since the last send.
func (s *RequestStreamState) ShouldSend(totalTokens int, finished bool) bool {
	if finished {
		return true
	}
	if s.SentTokens == 0 {
		return true
	}
	return (totalTokens - s.SentTokens) >= s.StreamInterval
}

// MarkSent records the token count at the time output was sent.
func (s *RequestStreamState) MarkSent(totalTokens int) {
	s.SentTokens = totalTokens
}
