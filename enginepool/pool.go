// SPDX-License-Identifier: MIT OR Apache-2.0

package enginepool

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Engine is the minimal seam the pool needs from a loaded engine: whether it
// has in-flight requests, so eviction never interrupts active generation.
type Engine interface {
	HasActiveRequests() bool
}

// ModelNotFoundError is returned when a requested model was never registered.
type ModelNotFoundError struct {
	ModelID   string
	Available []string
}

func (e *ModelNotFoundError) Error() string {
	avail := "(none)"
	if len(e.Available) > 0 {
		avail = strings.Join(e.Available, ", ")
	}
	return fmt.Sprintf("Model '%s' not found. Available models: %s", e.ModelID, avail)
}

// ModelTooLargeError is returned when a model alone exceeds the memory ceiling,
// so no amount of eviction can make it fit.
type ModelTooLargeError struct {
	ModelID   string
	ModelSize int64
	Ceiling   int64
}

func (e *ModelTooLargeError) Error() string {
	return fmt.Sprintf(
		"Model '%s' (%s) does not fit under the memory ceiling (%s). "+
			"Free system memory or lower memory_guard_tier.",
		e.ModelID, FormatSize(e.ModelSize), FormatSize(e.Ceiling))
}

// InsufficientMemoryError is returned when the model would fit on a clean
// process but the current usage leaves no room even after evicting everything
// evictable.
type InsufficientMemoryError struct {
	Required int64
	Current  int64
	Message  string
}

func (e *InsufficientMemoryError) Error() string { return e.Message }

// entry is the per-model state the pool tracks.
type entry struct {
	modelID       string
	modelPath     string
	estimatedSize int64
	lastAccess    float64
	pinned        bool
	engine        Engine // nil when not loaded
}

// Pool tracks registered models, their loaded engines, and LRU/pinning state.
// It is safe for concurrent bookkeeping, but a single logical admission
// (Admit) is expected to be serialized by the caller the way the reference
// holds its lock across get_engine.
type Pool struct {
	mu      sync.Mutex
	entries map[string]*entry
	order   []string // registration order, for deterministic listing
	now     func() float64
}

// New builds an empty pool. now supplies the monotonic-ish clock used for LRU
// timestamps (the reference uses time.time()); inject a controllable clock in
// tests.
func New(now func() float64) *Pool {
	return &Pool{entries: map[string]*entry{}, now: now}
}

// Register adds a model the pool can serve. It starts unloaded.
func (p *Pool) Register(modelID, modelPath string, estimatedSize int64, pinned bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.entries[modelID]; !ok {
		p.order = append(p.order, modelID)
	}
	p.entries[modelID] = &entry{
		modelID:       modelID,
		modelPath:     modelPath,
		estimatedSize: estimatedSize,
		pinned:        pinned,
	}
}

// Models lists registered model ids in registration order.
func (p *Pool) Models() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.order...)
}

// MarkLoaded records that model_id now holds a live engine and stamps its
// access time.
func (p *Pool) MarkLoaded(modelID string, eng Engine) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if e, ok := p.entries[modelID]; ok {
		e.engine = eng
		e.lastAccess = p.now()
	}
}

// MarkUnloaded clears the live engine for model_id.
func (p *Pool) MarkUnloaded(modelID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if e, ok := p.entries[modelID]; ok {
		e.engine = nil
	}
}

// Touch updates the access time of a loaded model (call on a cache hit).
func (p *Pool) Touch(modelID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if e, ok := p.entries[modelID]; ok {
		e.lastAccess = p.now()
	}
}

// IsLoaded reports whether model_id currently holds a live engine.
func (p *Pool) IsLoaded(modelID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	e, ok := p.entries[modelID]
	return ok && e.engine != nil
}

// Pin/Unpin toggle eviction protection for a model.
func (p *Pool) Pin(modelID string)   { p.setPinned(modelID, true) }
func (p *Pool) Unpin(modelID string) { p.setPinned(modelID, false) }

func (p *Pool) setPinned(modelID string, v bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if e, ok := p.entries[modelID]; ok {
		e.pinned = v
	}
}

// FindLRUVictim returns the least-recently-used loaded, non-pinned model that
// has no active requests, or ("", false) when nothing is evictable. Ties on the
// access timestamp are broken by model id, matching the reference's tuple sort.
func (p *Pool) FindLRUVictim() (string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.findLRUVictimLocked()
}

func (p *Pool) findLRUVictimLocked() (string, bool) {
	type cand struct {
		lastAccess float64
		modelID    string
	}
	var cands []cand
	for _, mid := range p.order {
		e := p.entries[mid]
		if e.engine == nil || e.pinned {
			continue
		}
		if e.engine.HasActiveRequests() {
			continue
		}
		cands = append(cands, cand{e.lastAccess, mid})
	}
	if len(cands) == 0 {
		return "", false
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].lastAccess != cands[j].lastAccess {
			return cands[i].lastAccess < cands[j].lastAccess
		}
		return cands[i].modelID < cands[j].modelID
	})
	return cands[0].modelID, true
}

// Admit runs the pre-load admission loop for model_id against a memory ceiling,
// mirroring the reference get_engine make-room logic. A ceiling <= 0 means the
// enforcer is off and the model is admitted unconditionally. Otherwise it
// repeatedly measures current memory via currentMemory; if the model would fit
// it returns nil, else it evicts the LRU victim via unload and retries. When
// nothing is evictable it returns ModelTooLargeError if the model alone exceeds
// the ceiling, otherwise InsufficientMemoryError.
//
// currentMemory and unload are the compute seams: in production they read
// max(active Metal memory, process footprint) and tear an engine down; the pool
// bookkeeping (clearing the evicted engine) is handled here so the next victim
// search is correct. Admit must be called under the caller's own serialization.
func (p *Pool) Admit(modelID string, ceiling int64, currentMemory func() int64, unload func(string)) error {
	p.mu.Lock()
	e, ok := p.entries[modelID]
	if !ok {
		order := append([]string(nil), p.order...)
		p.mu.Unlock()
		return &ModelNotFoundError{ModelID: modelID, Available: order}
	}
	estimated := e.estimatedSize
	p.mu.Unlock()

	if ceiling <= 0 {
		return nil
	}

	for {
		current := currentMemory()
		projected := current + estimated
		if projected <= ceiling {
			return nil
		}
		p.mu.Lock()
		victim, found := p.findLRUVictimLocked()
		p.mu.Unlock()
		if found {
			unload(victim)
			p.MarkUnloaded(victim)
			continue
		}
		if estimated > ceiling {
			return &ModelTooLargeError{ModelID: modelID, ModelSize: estimated, Ceiling: ceiling}
		}
		return &InsufficientMemoryError{
			Required: estimated,
			Current:  current,
			Message: fmt.Sprintf(
				"Cannot load %s: projected memory %s would exceed the memory ceiling %s "+
					"(current: %s, model: %s). Free system memory or lower memory_guard_tier.",
				modelID, FormatSize(projected), FormatSize(ceiling),
				FormatSize(current), FormatSize(estimated)),
		}
	}
}
