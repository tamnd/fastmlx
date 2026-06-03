// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

// ResolveBlockExtraKeys picks the cache-key salt that applies to a block ending
// at blockEnd, ported from resolve_block_extra_keys in paged_cache.py. It is the
// consumer of the per-request keying that VLMExtraKeysForCache /
// VLMExtraKeyTokenStartForCache / VLMExtraKeyRangesForCache produce: the result
// is mixed into the block hash so blocks that share tokens but cover different
// images never collide.
//
// Segmented ranges take precedence over a single extra-keys slice. Ranges must
// be sorted ascending by token start (VLMExtraKeyRangesForCache emits them in
// request order, which is sorted): the scan keeps the last range whose start is
// strictly before blockEnd and breaks on the first range at or after it, so an
// unsorted slice would resolve incorrectly. With no ranges, the single
// extra-keys slice applies when it is present (non-nil) and either has no token
// start or starts strictly before blockEnd. A nil extra-keys slice is the
// reference's None and yields nil.
func ResolveBlockExtraKeys(blockEnd int, extraKeys []string, extraKeyTokenStart *int, extraKeyRanges []VLMCacheKeySegment) []string {
	if len(extraKeyRanges) > 0 {
		var selected []string
		for _, r := range extraKeyRanges {
			if blockEnd > r.TokenStart {
				selected = r.ExtraKeys
			} else {
				break
			}
		}
		return selected
	}

	if extraKeys != nil && (extraKeyTokenStart == nil || blockEnd > *extraKeyTokenStart) {
		return extraKeys
	}
	return nil
}
