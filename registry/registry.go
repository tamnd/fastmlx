// SPDX-License-Identifier: MIT OR Apache-2.0

// Package registry tracks which engine currently owns each model, so that only
// one engine drives a shared model's batch generator at a time. The compute
// backend keeps per-model KV-cache state tied to the live model object; when two
// engines touch the same model that state becomes inconsistent. The registry is
// the GPU-free ownership state machine that guards against it.
//
// The model identity and the engine liveness are the seams. The reference keys
// the map on the live model object and holds the engine through a weak reference
// so a garbage-collected owner frees its slot automatically. Go has neither, so
// the caller supplies a stable string key for the model and an Owner whose
// Alive reports whether the engine is still around (the weak-reference probe) and
// whose Reset tears down the previous owner on a forced transfer (the scheduler
// deep-reset). Stale slots are reclaimed explicitly via IsOwned, Cleanup, and
// Stats rather than by the garbage collector.
package registry

import "sync"

// Owner is the engine side of an ownership slot. Alive reports whether the
// engine still exists (the reference's weakref() is not None); a dead owner is
// treated as no owner and its slot reclaimed. Reset tears the engine down on a
// forced transfer (the reference's owner.scheduler.deep_reset()); the caller's
// implementation absorbs the missing-attribute and error handling the reference
// swallows in _reset_owner.
type Owner interface {
	Alive() bool
	Reset()
}

// ModelOwnershipError is returned by Acquire when a live engine other than the
// caller already owns the model and force is not set.
type ModelOwnershipError struct {
	OwnerID string
}

func (e *ModelOwnershipError) Error() string {
	return "Model is already owned by engine " + e.OwnerID +
		". Use force=True or call release() on the previous engine first."
}

type slot struct {
	owner    Owner
	engineID string
}

// Stats is the registry's summary, matching the reference get_stats dict order.
type Stats struct {
	TotalEntries int `json:"total_entries"`
	ActiveOwners int `json:"active_owners"`
}

// Registry is the ownership map. The zero value is not usable; call New.
type Registry struct {
	mu     sync.Mutex
	owners map[string]slot
}

// New returns an empty registry.
func New() *Registry {
	return &Registry{owners: make(map[string]slot)}
}

// Acquire attempts to take ownership of the model named by key for engineID. If
// a different, still-alive engine already owns it, force transfers ownership
// (resetting the previous owner first) while the default path returns a
// ModelOwnershipError. A dead previous owner or a re-acquire by the same engine
// simply overwrites the slot. On success it returns true with a nil error.
func (r *Registry) Acquire(key string, owner Owner, engineID string, force bool) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.owners[key]; ok {
		if existing.owner.Alive() && existing.engineID != engineID {
			if force {
				existing.owner.Reset()
			} else {
				return false, &ModelOwnershipError{OwnerID: existing.engineID}
			}
		}
	}
	r.owners[key] = slot{owner: owner, engineID: engineID}
	return true, nil
}

// Release gives up ownership of key, but only when engineID is the recorded
// owner. It returns whether a slot was released.
func (r *Registry) Release(key, engineID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.owners[key]; ok && existing.engineID == engineID {
		delete(r.owners, key)
		return true
	}
	return false
}

// IsOwned reports whether key has a live owner and that owner's engine id. A
// slot whose owner is no longer alive is reclaimed and reported as unowned.
func (r *Registry) IsOwned(key string) (bool, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.owners[key]; ok {
		if existing.owner.Alive() {
			return true, existing.engineID
		}
		delete(r.owners, key)
	}
	return false, ""
}

// Cleanup drops every slot whose owner is no longer alive and returns how many
// were removed.
func (r *Registry) Cleanup() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	cleaned := 0
	for key, existing := range r.owners {
		if !existing.owner.Alive() {
			delete(r.owners, key)
			cleaned++
		}
	}
	return cleaned
}

// Stats reports the total slot count and how many have a live owner.
func (r *Registry) Stats() Stats {
	r.mu.Lock()
	defer r.mu.Unlock()
	active := 0
	for _, existing := range r.owners {
		if existing.owner.Alive() {
			active++
		}
	}
	return Stats{TotalEntries: len(r.owners), ActiveOwners: active}
}

var (
	defaultOnce sync.Once
	defaultReg  *Registry
)

// Default returns the process-wide registry, the analogue of the reference's
// module-level singleton. Tests should use New for isolated state.
func Default() *Registry {
	defaultOnce.Do(func() { defaultReg = New() })
	return defaultReg
}
