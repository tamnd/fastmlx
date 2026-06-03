// SPDX-License-Identifier: MIT OR Apache-2.0

package scheduler

import (
	"maps"
	"strconv"
	"sync"
)

// PrefillProgressTracker is the thread-safe, global per-request prefill progress
// store read by the admin stats API to show per-request prefill progress in the
// Active Models card. It is fed once per prefill chunk by the generator's
// progress callback (CPU counters only), distinct from the per-model
// PrefillTransientTracker that drives the memory throttle.
//
// Each entry holds a request's processed/total token counts, model id, phase,
// and timing. An entry is auto-removed once processed reaches total. The
// monotonic clock the reference reads internally is injected here as a now
// value (seconds) so the logic stays deterministic and testable.
type PrefillProgressTracker struct {
	mu      sync.Mutex
	order   []string // request ids in insertion order, for stable result ordering
	entries map[string]*progressEntry
}

type progressEntry struct {
	processed int
	total     int
	modelID   string
	startTime float64
	lastTime  float64
	speed     float64
	phase     string
	detail    *string
	extra     map[string]float64
}

// extraKeyOrder is the fixed set of speculative-prefill detail fields surfaced
// from an entry's extra payload, emitted in this order after the standard
// fields (matching the reference's tuple).
var extraKeyOrder = []string{
	"scored_tokens",
	"selected_tokens",
	"keep_percent",
	"prompt_tokens",
	"system_tokens",
	"conversation_tokens",
	"cached_tokens",
}

// ProgressResult is one prefilling request's view for the dashboard.
type ProgressResult struct {
	RequestID string
	Processed int
	Total     int
	Speed     float64
	ETA       *float64 // nil when speed is not positive
	Elapsed   float64
	Phase     string
	Detail    *string // nil when no detail was set
	Extra     []ExtraField
}

// ExtraField is one passed-through speculative-prefill detail value. The
// reference's extras are all numeric, so they are carried as float64.
type ExtraField struct {
	Key   string
	Value float64
}

// NewPrefillProgressTracker returns an empty tracker.
func NewPrefillProgressTracker() *PrefillProgressTracker {
	return &PrefillProgressTracker{entries: map[string]*progressEntry{}}
}

// Update records prefill progress for a request, auto-removing the entry once
// processed reaches total. A phase change resets the entry's start time and
// speed. now is the injected monotonic clock in seconds. detail is nil for no
// detail; extra carries optional speculative-prefill fields (nil for none).
func (t *PrefillProgressTracker) Update(requestID string, processed, total int, modelID, phase string, detail *string, extra map[string]float64, now float64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if processed >= total {
		t.popLocked(requestID)
		return
	}

	prev := t.entries[requestID]
	phaseChanged := prev != nil && prev.phase != phase

	startTime := now
	if prev != nil && !phaseChanged {
		startTime = prev.startTime
	}

	speed := 0.0
	if prev != nil && !phaseChanged {
		dt := now - prev.lastTime
		dtok := processed - prev.processed
		if dt > 0 && dtok > 0 {
			speed = float64(dtok) / dt
		} else {
			speed = prev.speed
		}
	}

	e := &progressEntry{
		processed: processed,
		total:     total,
		modelID:   modelID,
		startTime: startTime,
		lastTime:  now,
		speed:     speed,
		phase:     phase,
		detail:    detail,
		extra:     copyExtra(extra),
	}
	if prev == nil {
		t.order = append(t.order, requestID)
	}
	t.entries[requestID] = e
}

// UpdatePrefill is Update for the default "prefill" phase with no detail or
// extra payload.
func (t *PrefillProgressTracker) UpdatePrefill(requestID string, processed, total int, modelID string, now float64) {
	t.Update(requestID, processed, total, modelID, "prefill", nil, nil, now)
}

// Remove explicitly drops a request, e.g. on abort or finish.
func (t *PrefillProgressTracker) Remove(requestID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.popLocked(requestID)
}

// GetModelProgress returns the prefilling requests for a model in insertion
// order. now is the injected monotonic clock used to derive elapsed time.
func (t *PrefillProgressTracker) GetModelProgress(modelID string, now float64) []ProgressResult {
	t.mu.Lock()
	defer t.mu.Unlock()

	var results []ProgressResult
	for _, rid := range t.order {
		e := t.entries[rid]
		if e.modelID != modelID {
			continue
		}
		elapsed := now - e.startTime
		remaining := e.total - e.processed
		var eta *float64
		if e.speed > 0 {
			v := round1(float64(remaining) / e.speed)
			eta = &v
		}
		res := ProgressResult{
			RequestID: rid,
			Processed: e.processed,
			Total:     e.total,
			Speed:     round1(e.speed),
			ETA:       eta,
			Elapsed:   round1(elapsed),
			Phase:     e.phase,
			Detail:    e.detail,
		}
		for _, key := range extraKeyOrder {
			if v, ok := e.extra[key]; ok {
				res.Extra = append(res.Extra, ExtraField{Key: key, Value: v})
			}
		}
		results = append(results, res)
	}
	return results
}

// Clear removes all entries.
func (t *PrefillProgressTracker) Clear() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.order = nil
	t.entries = map[string]*progressEntry{}
}

// popLocked removes a request from the entry map and order slice. The caller
// holds the lock.
func (t *PrefillProgressTracker) popLocked(requestID string) {
	if _, ok := t.entries[requestID]; !ok {
		return
	}
	delete(t.entries, requestID)
	for i, rid := range t.order {
		if rid == requestID {
			t.order = append(t.order[:i], t.order[i+1:]...)
			break
		}
	}
}

func copyExtra(extra map[string]float64) map[string]float64 {
	if len(extra) == 0 {
		return nil
	}
	out := make(map[string]float64, len(extra))
	maps.Copy(out, extra)
	return out
}

// round1 rounds to one decimal place the way Python's round() does, by
// formatting the true binary value rather than the x*10 trick.
func round1(x float64) float64 {
	f, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 1, 64), 64)
	return f
}

var (
	prefillTrackerOnce sync.Once
	prefillTracker     *PrefillProgressTracker
)

// GetPrefillTracker returns the lazily created global tracker singleton.
func GetPrefillTracker() *PrefillProgressTracker {
	prefillTrackerOnce.Do(func() {
		prefillTracker = NewPrefillProgressTracker()
	})
	return prefillTracker
}
