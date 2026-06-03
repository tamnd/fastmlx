// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import "math"

// This file holds the pure per-request timing math of the active-models builder:
// the generating-entry derivation, the waiting-entry shaping, the loading-time
// estimate, and the idle/TTL remaining. The request objects and the clocks are
// caller seams, so the timestamps, counts, and the resolved effective TTL all
// arrive as inputs. An optional timestamp is nil for Python's None; a zero
// timestamp is falsy and treated as absent, matching the `if started_at` guards.

const bytesPerGB = 1 << 30

// toFloat coerces a numeric value to float64, returning 0 for anything else.
func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case bool:
		if n {
			return 1
		}
		return 0
	default:
		return 0
	}
}

// BuildGeneratingEntry derives the dashboard entry for an actively generating
// request. The elapsed time and the last-activity age are nil when their start
// timestamp is absent or zero; the token rate is the generated tokens over the
// elapsed time, or zero when no positive time has passed.
func BuildGeneratingEntry(now float64, requestID string, startedAt, lastActivityAt any, generatedTokens, promptTokens int, maxTokens any) map[string]any {
	var elapsed any
	if pyTruthy(startedAt) {
		elapsed = math.Max(0.0, now-toFloat(startedAt))
	}
	var lastActivityAge any
	if pyTruthy(lastActivityAt) {
		lastActivityAge = math.Max(0.0, now-toFloat(lastActivityAt))
	}
	tokensPerSecond := 0.0
	if e, ok := elapsed.(float64); ok && e > 0 {
		tokensPerSecond = float64(generatedTokens) / e
	}
	return map[string]any{
		"request_id":                requestID,
		"elapsed_seconds":           elapsed,
		"generated_tokens":          generatedTokens,
		"tokens_per_second":         tokensPerSecond,
		"last_activity_age_seconds": lastActivityAge,
		"prompt_tokens":             promptTokens,
		"max_tokens":                maxTokens,
	}
}

// BuildWaitingEntry shapes the dashboard entry for a queued request. The queue
// position is its one-based place in the queue and the elapsed time is how long
// it has waited since arrival, floored at zero.
func BuildWaitingEntry(now float64, requestID string, queuePosition int, arrivalTime float64, promptTokens int) map[string]any {
	return map[string]any{
		"request_id":      requestID,
		"queue_position":  queuePosition,
		"elapsed_seconds": math.Max(0.0, now-arrivalTime),
		"prompt_tokens":   promptTokens,
	}
}

// LoadingEstimate derives the loading-progress estimate for a model still
// loading. The elapsed time is nil when the start timestamp is absent or zero.
// An estimate is produced only once a per-GB load rate has been observed across
// more than one sample, floored at three seconds; the remaining estimate is the
// gap to that estimate while the elapsed time is still under it.
func LoadingEstimate(now float64, loadingStartedAt any, estimatedSize int, observedSecondsPerGB any, observations int) map[string]any {
	var elapsed any
	if pyTruthy(loadingStartedAt) {
		elapsed = math.Max(0.0, now-toFloat(loadingStartedAt))
	}
	var estimated any
	var remaining any
	if e, ok := elapsed.(float64); ok {
		sizeGB := float64(estimatedSize) / float64(bytesPerGB)
		if pyTruthy(observedSecondsPerGB) && observations >= 2 {
			est := math.Max(3.0, 1.0+sizeGB*toFloat(observedSecondsPerGB))
			estimated = est
			if e < est {
				remaining = math.Max(0.0, est-e)
			}
		}
	}
	return map[string]any{
		"loading_elapsed_seconds":            elapsed,
		"loading_estimated_seconds":          estimated,
		"loading_remaining_seconds_estimate": remaining,
	}
}

// IdleAndTTL derives how long a loaded model has been idle and how much of its
// TTL remains. Both are nil unless the model is loaded; the idle time also
// needs a positive last-access timestamp, and the TTL remaining needs both a
// resolved effective TTL and a known idle time, floored at zero.
func IdleAndTTL(now float64, isLoaded bool, lastAccess, effectiveTTL any) map[string]any {
	var idle any
	if isLoaded && lastAccess != nil && toFloat(lastAccess) > 0 {
		idle = math.Max(0.0, now-toFloat(lastAccess))
	}
	var ttlRemaining any
	if isLoaded && effectiveTTL != nil {
		if i, ok := idle.(float64); ok {
			ttlRemaining = math.Max(0.0, toFloat(effectiveTTL)-i)
		}
	}
	return map[string]any{
		"idle_seconds":          idle,
		"ttl_remaining_seconds": ttlRemaining,
	}
}
