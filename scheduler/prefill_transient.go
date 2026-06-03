// SPDX-License-Identifier: MIT OR Apache-2.0

// Package scheduler holds the request-scheduling logic. This file ports the
// per-scheduler estimator of prefill chunk transient memory; the engine-driven
// loop that consumes it stays compute-gated for now.
package scheduler

// PrefillTransientTracker is an EWMA estimator of the transient bytes a prefill
// chunk allocates per token. Each loaded model owns one. The adaptive prefill
// throttle sizes the next chunk so its predicted transient stays under the
// remaining memory headroom; until a sample has been recorded the caller falls
// back to a static estimate.
type PrefillTransientTracker struct {
	modelID       string
	ewmaPerToken  float64
	samples       int
	lastDeltaByte int
	lastNTokens   int
}

// ewmaAlpha is the weight on the most recent chunk.
const ewmaAlpha = 0.3

// NewPrefillTransientTracker returns a tracker for the given model id.
func NewPrefillTransientTracker(modelID string) *PrefillTransientTracker {
	return &PrefillTransientTracker{modelID: modelID}
}

// Update records one chunk observation. Non-positive token counts or transient
// deltas are skipped: a non-positive delta means the MLX cache pool reclaimed
// more than this chunk allocated, and folding it in would bias the EWMA toward
// zero and underestimate the next chunk.
func (t *PrefillTransientTracker) Update(nTokens, transientBytes int) {
	if nTokens <= 0 {
		return
	}
	if transientBytes <= 0 {
		return
	}

	perToken := float64(transientBytes) / float64(nTokens)
	if t.samples == 0 {
		t.ewmaPerToken = perToken
	} else {
		t.ewmaPerToken = ewmaAlpha*perToken + (1.0-ewmaAlpha)*t.ewmaPerToken
	}
	t.samples++
	t.lastDeltaByte = transientBytes
	t.lastNTokens = nTokens
}

// Predict returns the predicted transient bytes for a chunk of nTokens, scaled
// by safetyFactor. It returns 0 when no samples have been recorded yet or when
// nTokens is non-positive, in which case the caller must use a static estimate.
// The reference passes safety_factor=1.2 by default; PredictDefault applies it.
func (t *PrefillTransientTracker) Predict(nTokens int, safetyFactor float64) int {
	if t.samples == 0 || nTokens <= 0 {
		return 0
	}
	return int(t.ewmaPerToken * float64(nTokens) * safetyFactor)
}

// PredictDefault is Predict with the reference's default safety factor of 1.2.
func (t *PrefillTransientTracker) PredictDefault(nTokens int) int {
	return t.Predict(nTokens, 1.2)
}

// BytesPerToken is the current EWMA value (bytes per prefill token), 0 if no
// samples have been recorded.
func (t *PrefillTransientTracker) BytesPerToken() float64 { return t.ewmaPerToken }

// Samples is the number of chunks recorded since the last reset.
func (t *PrefillTransientTracker) Samples() int { return t.samples }

// LastDeltaBytes is the bytes added by the most recently measured chunk.
func (t *PrefillTransientTracker) LastDeltaBytes() int { return t.lastDeltaByte }

// LastNTokens is the token count of the most recently measured chunk.
func (t *PrefillTransientTracker) LastNTokens() int { return t.lastNTokens }

// Reset drops all observations, e.g. on model reload or after a long idle.
func (t *PrefillTransientTracker) Reset() {
	t.ewmaPerToken = 0.0
	t.samples = 0
	t.lastDeltaByte = 0
	t.lastNTokens = 0
}
