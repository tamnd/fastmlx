// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

// This file ports the per-request derivations the prefix cache uses to key a
// VLM request on its images. A request that carries images contributes its
// image hash (or per-image-turn hashes) as extra keys mixed into the block hash,
// so two requests with the same tokens but different images never share cached
// blocks. These are pure derivations off the request's already-computed VLM
// fields; computing those fields (decoding and hashing the images) is upstream.

// VLMCacheKeyRange is one entry of a request's segmented VLM cache keying: the
// token index where an image turn begins and that image's hash.
type VLMCacheKeyRange struct {
	TokenStart int
	ImageHash  string
}

// VLMCacheKeySegment is a resolved cache-key range: the token start paired with
// the extra keys that apply from there on. ExtraKeys is a one-element slice
// matching the reference's single-hash tuple, kept as a slice so it composes
// with the multi-key block-hash path.
type VLMCacheKeySegment struct {
	TokenStart int
	ExtraKeys  []string
}

// VLMExtraKeysForCache wraps a whole-request image hash as the extra-keys slice
// the block hasher mixes in, or returns nil when the request has no images.
func VLMExtraKeysForCache(vlmImageHash string) []string {
	if vlmImageHash != "" {
		return []string{vlmImageHash}
	}
	return nil
}

// VLMExtraKeyTokenStartForCache reports the token index where image-specific
// cache keying begins, valid only when the request carries an image hash. The
// boolean is false (and the index meaningless) for a request with no images.
func VLMExtraKeyTokenStartForCache(vlmImageHash string, vlmCacheKeyStart int) (int, bool) {
	if vlmImageHash != "" {
		return vlmCacheKeyStart, true
	}
	return 0, false
}

// VLMExtraKeyRangesForCache resolves the segmented per-image-turn cache keying:
// each range's start is paired with that image's hash wrapped as a one-element
// extra-keys slice. It returns nil when the request has no segmented ranges.
func VLMExtraKeyRangesForCache(ranges []VLMCacheKeyRange) []VLMCacheKeySegment {
	if len(ranges) == 0 {
		return nil
	}
	out := make([]VLMCacheKeySegment, len(ranges))
	for i, r := range ranges {
		out[i] = VLMCacheKeySegment{TokenStart: r.TokenStart, ExtraKeys: []string{r.ImageHash}}
	}
	return out
}
