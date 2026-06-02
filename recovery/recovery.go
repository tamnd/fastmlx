// SPDX-License-Identifier: MIT OR Apache-2.0

// Package recovery holds the GPU-free half of the cache-recovery surface: the
// classifier that decides whether an error coming out of a decode step is the
// kind of cache corruption the scheduler can recover from by clearing caches
// and rescheduling its running requests.
//
// The recovery orchestration itself (tearing down the batch generator, clearing
// the block-aware cache, moving running requests back to the waiting queue, then
// forcing a GC) lives on the live decode path and is wired in once the compute
// backend lands. The pattern match below is the portable, self-contained piece:
// it only inspects an error's message text.
package recovery

import "strings"

// CorruptionPatterns are the message substrings that mark an error as cache
// corruption. They mirror the failure modes seen on the heterogeneous-batch
// decode path: a None KV-cache slot subscripted or iterated, a missing cache
// attribute, or a shape/broadcast mismatch in attention. The substrings are
// matched verbatim, in the same order as upstream.
var CorruptionPatterns = []string{
	"'NoneType' object is not subscriptable",
	"'NoneType' object is not iterable",
	"BatchKVCache",
	"KVCache",
	"cache.keys",
	"cache.values",
	"'NoneType' object has no attribute",
	"not broadcastable",
	"cannot be broadcast",
	"shape mismatch",
}

// IsCorruptionMessage reports whether a message contains any cache-corruption
// pattern. It is the message-only form, useful when the failure arrives as a
// string rather than an error value.
func IsCorruptionMessage(msg string) bool {
	for _, p := range CorruptionPatterns {
		if strings.Contains(msg, p) {
			return true
		}
	}
	return false
}

// IsCorruptionError reports whether an error looks like cache corruption,
// matching its Error() text against CorruptionPatterns. A nil error is never
// corruption.
func IsCorruptionError(err error) bool {
	if err == nil {
		return false
	}
	return IsCorruptionMessage(err.Error())
}
