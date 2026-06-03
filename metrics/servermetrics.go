// SPDX-License-Identifier: MIT OR Apache-2.0

// Package metrics tracks server-level token and throughput counters for the
// status dashboard. It accumulates session and all-time totals (plus a per-model
// breakdown) and derives a snapshot of served tokens, cache efficiency, and
// average prefill/generation throughput.
package metrics

import (
	"strconv"
	"sync"
)

// counters holds the six raw totals tracked per scope and per model.
type counters struct {
	promptTokens       int
	completionTokens   int
	cachedTokens       int
	requests           int
	prefillDuration    float64
	generationDuration float64
}

func (c *counters) add(prompt, completion, cached int, prefill, generation float64) {
	c.promptTokens += prompt
	c.completionTokens += completion
	c.cachedTokens += cached
	c.requests++
	c.prefillDuration += prefill
	c.generationDuration += generation
}

// ServerMetrics is the thread-safe global counter store. The all-time totals are
// persisted across restarts in the reference; that disk seam (load/save) is
// deferred, so this type holds the in-memory accumulation and snapshot logic.
type ServerMetrics struct {
	mu              sync.Mutex
	session         counters
	alltime         counters
	perModel        map[string]*counters
	alltimePerModel map[string]*counters
}

// NewServerMetrics returns an empty metrics store.
func NewServerMetrics() *ServerMetrics {
	return &ServerMetrics{
		perModel:        map[string]*counters{},
		alltimePerModel: map[string]*counters{},
	}
}

// Snapshot is a derived metrics view for one scope/model.
type Snapshot struct {
	TotalTokensServed     int     `json:"total_tokens_served"`
	TotalCachedTokens     int     `json:"total_cached_tokens"`
	CacheEfficiency       float64 `json:"cache_efficiency"`
	TotalPromptTokens     int     `json:"total_prompt_tokens"`
	TotalCompletionTokens int     `json:"total_completion_tokens"`
	TotalRequests         int     `json:"total_requests"`
	AvgPrefillTPS         float64 `json:"avg_prefill_tps"`
	AvgGenerationTPS      float64 `json:"avg_generation_tps"`
	UptimeSeconds         float64 `json:"uptime_seconds"`
}

// RecordRequestComplete adds one completed request's tokens and durations to the
// session and all-time totals, and to the per-model breakdown when a model id is
// given. Thread-safe. Periodic persistence is deferred (disk seam).
func (s *ServerMetrics) RecordRequestComplete(promptTokens, completionTokens, cachedTokens int, prefillDuration, generationDuration float64, modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.session.add(promptTokens, completionTokens, cachedTokens, prefillDuration, generationDuration)
	s.alltime.add(promptTokens, completionTokens, cachedTokens, prefillDuration, generationDuration)

	if modelID != "" {
		modelCounters(s.perModel, modelID).add(promptTokens, completionTokens, cachedTokens, prefillDuration, generationDuration)
		modelCounters(s.alltimePerModel, modelID).add(promptTokens, completionTokens, cachedTokens, prefillDuration, generationDuration)
	}
}

func modelCounters(m map[string]*counters, modelID string) *counters {
	c, ok := m[modelID]
	if !ok {
		c = &counters{}
		m[modelID] = c
	}
	return c
}

// GetSnapshot returns the metrics snapshot for a scope ("session" or "alltime")
// and optional model id. An unknown model yields a zero snapshot carrying the
// given uptime. uptime is the injected clock seam (the reference's
// now - start_time). Thread-safe.
func (s *ServerMetrics) GetSnapshot(modelID, scope string, uptime float64) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	global := &s.session
	perModel := s.perModel
	if scope == "alltime" {
		global = &s.alltime
		perModel = s.alltimePerModel
	}

	c := global
	if modelID != "" {
		m, ok := perModel[modelID]
		if !ok {
			return buildSnapshot(counters{}, uptime)
		}
		c = m
	}
	return buildSnapshot(*c, uptime)
}

// buildSnapshot derives the snapshot from raw counters. actual_processed strips
// cached tokens from the prompt count for the prefill rate; rates and cache
// efficiency are zero when their denominator is zero. All derived floats are
// rounded to one decimal the way Python's round does.
func buildSnapshot(c counters, uptime float64) Snapshot {
	actualProcessed := c.promptTokens - c.cachedTokens
	avgPrefillTPS := 0.0
	if c.prefillDuration > 0 {
		avgPrefillTPS = float64(actualProcessed) / c.prefillDuration
	}
	avgGenerationTPS := 0.0
	if c.generationDuration > 0 {
		avgGenerationTPS = float64(c.completionTokens) / c.generationDuration
	}
	cacheEfficiency := 0.0
	if c.promptTokens > 0 {
		cacheEfficiency = float64(c.cachedTokens) / float64(c.promptTokens) * 100
	}
	return Snapshot{
		TotalTokensServed:     c.promptTokens + c.completionTokens,
		TotalCachedTokens:     c.cachedTokens,
		CacheEfficiency:       round1(cacheEfficiency),
		TotalPromptTokens:     c.promptTokens,
		TotalCompletionTokens: c.completionTokens,
		TotalRequests:         c.requests,
		AvgPrefillTPS:         round1(avgPrefillTPS),
		AvgGenerationTPS:      round1(avgGenerationTPS),
		UptimeSeconds:         round1(uptime),
	}
}

// round1 rounds to one decimal place exactly as Python's round() does: by
// correctly rounding the true binary value (ties to even), which strconv's
// decimal formatting performs. The naive x*10 then round trick diverges on
// values like 0.05, so it is avoided.
func round1(x float64) float64 {
	f, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 1, 64), 64)
	return f
}
