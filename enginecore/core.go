// SPDX-License-Identifier: Apache-2.0

// Package enginecore drives the scheduler from a single goroutine and fans
// per-request increments out over channels. It replaces an asyncio-style engine
// loop with goroutines and channels - no event loop, no GIL - // and houses the concrete BatchedEngine that the HTTP routes talk to.
package enginecore

import (
	"context"
	"sync"
	"time"

	"github.com/tamnd/fastmlx/engine"
	"github.com/tamnd/fastmlx/scheduler"
)

// ErrQueueFull and ErrNotRunning re-export the engine sentinels so callers that
// only import enginecore keep working; the HTTP layer maps ErrQueueFull to 503.
var (
	ErrQueueFull  = engine.ErrQueueFull
	ErrNotRunning = engine.ErrNotRunning
)

// idleTick bounds how long the loop blocks when idle so Stop is responsive even
// without new work.
const idleTick = 50 * time.Millisecond

// outChan is a per-request delivery slot. The loop never blocks on a slow client:
// when the buffered channel is full it coalesces the backlog (cumulative fields
// supersede, deltas concatenate) and retries on later steps - the output_collector
// smart-aggregation behavior under backpressure.
type outChan struct {
	ch         chan engine.RequestOutput
	pending    engine.RequestOutput
	hasPending bool
	closed     bool
}

// Core owns the scheduler and the loop goroutine.
type Core struct {
	sched         *scheduler.Scheduler
	maxConcurrent int

	submitCh chan *engine.Request
	wake     chan struct{}

	mu       sync.Mutex
	outs     map[string]*outChan
	inFlight int
	running  bool

	loopDone chan struct{}
}

// NewCore builds a core over a scheduler. maxConcurrent <= 0 disables admission
// control.
func NewCore(sched *scheduler.Scheduler, maxConcurrent int) *Core {
	return &Core{
		sched:         sched,
		maxConcurrent: maxConcurrent,
		submitCh:      make(chan *engine.Request, 64),
		wake:          make(chan struct{}, 1),
		outs:          make(map[string]*outChan),
	}
}

// Start launches the loop goroutine. It stops when ctx is cancelled.
func (c *Core) Start(ctx context.Context) {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return
	}
	c.running = true
	c.loopDone = make(chan struct{})
	c.mu.Unlock()
	go c.loop(ctx)
}

// Stop waits for the loop to drain and exit. Cancel the context passed to Start
// to trigger shutdown.
func (c *Core) Stop() {
	c.mu.Lock()
	done := c.loopDone
	running := c.running
	c.mu.Unlock()
	if running && done != nil {
		<-done
	}
}

// Submit admits a request and returns the channel its increments stream over.
// The channel is closed when the request finishes (or is aborted/errored).
func (c *Core) Submit(req *engine.Request) (<-chan engine.RequestOutput, error) {
	c.mu.Lock()
	if !c.running {
		c.mu.Unlock()
		return nil, ErrNotRunning
	}
	if c.maxConcurrent > 0 && c.inFlight >= c.maxConcurrent {
		c.mu.Unlock()
		return nil, ErrQueueFull
	}
	c.inFlight++
	oc := &outChan{ch: make(chan engine.RequestOutput, 64)}
	c.outs[req.ID] = oc
	c.mu.Unlock()

	c.submitCh <- req
	c.signal()
	return oc.ch, nil
}

// Abort requests cancellation of an in-flight request.
func (c *Core) Abort(id string) {
	c.sched.Abort(id)
	c.signal()
}

// InFlight reports the current admitted request count.
func (c *Core) InFlight() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inFlight
}

func (c *Core) signal() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *Core) loop(ctx context.Context) {
	defer close(c.loopDone)
	defer c.shutdown()

	for {
		// Drain newly submitted requests into the scheduler.
		c.drainSubmits()

		hasPending := c.flushPending()
		work := c.sched.HasUnfinished()

		if !work && !hasPending {
			// Idle: wait for new work, a wake signal, or shutdown.
			select {
			case <-ctx.Done():
				return
			case req := <-c.submitCh:
				c.sched.AddRequest(req)
			case <-c.wake:
			case <-time.After(idleTick):
			}
			continue
		}

		select {
		case <-ctx.Done():
			return
		default:
		}

		if work {
			out, _ := c.sched.Step()
			for i := range out.Outputs {
				c.deliver(out.Outputs[i])
			}
		}
	}
}

func (c *Core) drainSubmits() {
	for {
		select {
		case req := <-c.submitCh:
			c.sched.AddRequest(req)
		default:
			return
		}
	}
}

// deliver routes one increment to its request channel, coalescing on backpressure.
func (c *Core) deliver(o engine.RequestOutput) {
	c.mu.Lock()
	oc, ok := c.outs[o.RequestID]
	if !ok || oc.closed {
		c.mu.Unlock()
		return
	}
	if oc.hasPending {
		o = coalesce(oc.pending, o)
	}
	select {
	case oc.ch <- o:
		oc.hasPending = false
		oc.pending = engine.RequestOutput{}
		if o.Finished {
			c.closeLocked(oc, o.RequestID)
		}
	default:
		oc.pending = o
		oc.hasPending = true
	}
	c.mu.Unlock()
}

// flushPending retries coalesced sends. Returns true if any backlog remains.
func (c *Core) flushPending() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	remaining := false
	for id, oc := range c.outs {
		if oc.closed || !oc.hasPending {
			continue
		}
		select {
		case oc.ch <- oc.pending:
			fin := oc.pending.Finished
			oc.hasPending = false
			oc.pending = engine.RequestOutput{}
			if fin {
				c.closeLocked(oc, id)
			}
		default:
			remaining = true
		}
	}
	return remaining
}

// closeLocked finalizes a request channel. Caller holds c.mu.
func (c *Core) closeLocked(oc *outChan, id string) {
	if oc.closed {
		return
	}
	oc.closed = true
	close(oc.ch)
	delete(c.outs, id)
	if c.inFlight > 0 {
		c.inFlight--
	}
}

// shutdown closes any open request channels so blocked consumers unblock.
func (c *Core) shutdown() {
	c.mu.Lock()
	for id, oc := range c.outs {
		if !oc.closed {
			oc.closed = true
			close(oc.ch)
		}
		delete(c.outs, id)
	}
	c.inFlight = 0
	c.running = false
	c.mu.Unlock()
}

// coalesce merges a pending increment with a newer one: cumulative fields take
// the newer value, the streamed delta is the concatenation.
func coalesce(prev, next engine.RequestOutput) engine.RequestOutput {
	next.NewText = prev.NewText + next.NewText
	next.NewTokenIDs = append(append([]int(nil), prev.NewTokenIDs...), next.NewTokenIDs...)
	return next
}
