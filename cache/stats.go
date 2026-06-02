// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

// Cache statistics shared across the cache tiers (prefix, paged, VLM, paged
// SSD). These are pure counters with a handful of derived ratios; the cache
// cores call the record* methods and the admin stats endpoint serializes a
// snapshot. Each Snapshot returns the fields in the reference key order so the
// dashboard sees a stable shape.

// BaseStats holds the counters every cache tier shares.
type BaseStats struct {
	Hits      int
	Misses    int
	Evictions int
}

// TotalQueries is hits plus misses.
func (s *BaseStats) TotalQueries() int { return s.Hits + s.Misses }

// HitRate is hits over total queries, 0 when there have been none.
func (s *BaseStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0.0
	}
	return float64(s.Hits) / float64(total)
}

// RecordHit, RecordMiss, and RecordEviction bump the shared counters.
func (s *BaseStats) RecordHit()      { s.Hits++ }
func (s *BaseStats) RecordMiss()     { s.Misses++ }
func (s *BaseStats) RecordEviction() { s.Evictions++ }

// Reset zeroes the shared counters.
func (s *BaseStats) Reset() {
	s.Hits = 0
	s.Misses = 0
	s.Evictions = 0
}

func (s *BaseStats) baseDict() []kv {
	return []kv{
		{"hits", s.Hits},
		{"misses", s.Misses},
		{"evictions", s.Evictions},
	}
}

// PrefixStats tracks prefix-cache reuse: tokens saved and partial-block skip
// metrics for observability.
type PrefixStats struct {
	BaseStats
	TokensSaved              int
	PartialBlockSkips        int
	PartialTokensSkipped     int
	BlockSize                int
	LastPartialTokensSkipped int
	LastTokensToNextBlock    int
	TokensMatchedTotal       int
	TokensRequestedTotal     int
	totalQueries             int
}

// TotalQueries uses the explicit counter when it has been set, else hits+misses.
func (s *PrefixStats) TotalQueries() int {
	if s.totalQueries > 0 {
		return s.totalQueries
	}
	return s.Hits + s.Misses
}

// SetTotalQueries sets the explicit query counter (legacy compatibility).
func (s *PrefixStats) SetTotalQueries(v int) { s.totalQueries = v }

// HitRate divides hits by the prefix total-query count, which honors the
// explicit counter when set (the reference hit_rate reads total_queries, and
// PrefixCacheStats overrides it).
func (s *PrefixStats) HitRate() float64 {
	total := s.TotalQueries()
	if total == 0 {
		return 0.0
	}
	return float64(s.Hits) / float64(total)
}

// Reset zeroes all prefix statistics.
func (s *PrefixStats) Reset() {
	s.BaseStats.Reset()
	s.TokensSaved = 0
	s.PartialBlockSkips = 0
	s.PartialTokensSkipped = 0
	s.LastPartialTokensSkipped = 0
	s.LastTokensToNextBlock = 0
	s.TokensMatchedTotal = 0
	s.TokensRequestedTotal = 0
	s.totalQueries = 0
}

// Snapshot renders the prefix stats in the reference key order.
func (s *PrefixStats) Snapshot() string {
	pairs := s.baseDict()
	pairs = append(pairs,
		kv{"tokens_saved", s.TokensSaved},
		kv{"partial_block_skips", s.PartialBlockSkips},
		kv{"partial_tokens_skipped", s.PartialTokensSkipped},
		kv{"block_size", s.BlockSize},
		kv{"last_partial_tokens_skipped", s.LastPartialTokensSkipped},
		kv{"last_tokens_to_next_block", s.LastTokensToNextBlock},
		kv{"tokens_matched_total", s.TokensMatchedTotal},
		kv{"tokens_requested_total", s.TokensRequestedTotal},
		kv{"_total_queries", s.totalQueries},
		kv{"total_queries", s.TotalQueries()},
		kv{"hit_rate", s.HitRate()},
	)
	return encodeOrdered(pairs)
}

// PagedStats tracks the paged KV cache with block-level metrics.
type PagedStats struct {
	BaseStats
	TotalBlocks       int
	AllocatedBlocks   int
	FreeBlocks        int
	SharedBlocks      int // blocks with ref_count > 1
	TotalTokensCached int
	COWCopies         int // copy-on-write operations
}

// Utilization is allocated over total blocks, 0 when there are no blocks.
func (s *PagedStats) Utilization() float64 {
	if s.TotalBlocks == 0 {
		return 0.0
	}
	return float64(s.AllocatedBlocks) / float64(s.TotalBlocks)
}

// Reset zeroes the runtime statistics, leaving the capacity metrics.
func (s *PagedStats) Reset() {
	s.BaseStats.Reset()
	s.COWCopies = 0
}

// Snapshot renders the paged stats in the reference key order.
func (s *PagedStats) Snapshot() string {
	pairs := s.baseDict()
	pairs = append(pairs,
		kv{"total_blocks", s.TotalBlocks},
		kv{"allocated_blocks", s.AllocatedBlocks},
		kv{"free_blocks", s.FreeBlocks},
		kv{"shared_blocks", s.SharedBlocks},
		kv{"total_tokens_cached", s.TotalTokensCached},
		kv{"cow_copies", s.COWCopies},
		kv{"total_queries", s.TotalQueries()},
		kv{"hit_rate", s.HitRate()},
		kv{"utilization", s.Utilization()},
	)
	return encodeOrdered(pairs)
}

// VLMStats tracks the vision-language cache with image-reuse metrics.
type VLMStats struct {
	BaseStats
	TokensSaved    int
	ImageCacheHits int
}

// RecordImageHit records an image cache hit.
func (s *VLMStats) RecordImageHit() { s.ImageCacheHits++ }

// Reset zeroes all VLM statistics.
func (s *VLMStats) Reset() {
	s.BaseStats.Reset()
	s.TokensSaved = 0
	s.ImageCacheHits = 0
}

// Snapshot renders the VLM stats in the reference key order.
func (s *VLMStats) Snapshot() string {
	pairs := s.baseDict()
	pairs = append(pairs,
		kv{"tokens_saved", s.TokensSaved},
		kv{"image_cache_hits", s.ImageCacheHits},
		kv{"total_queries", s.TotalQueries()},
		kv{"hit_rate", s.HitRate()},
	)
	return encodeOrdered(pairs)
}

// PagedSSDStats tracks the cold SSD tier: operation counters, storage capacity,
// and the in-memory hot-tier metrics.
type PagedSSDStats struct {
	BaseStats
	// Operation counters.
	Saves         int
	Loads         int
	Errors        int
	SSDWriteDrops int
	// Storage capacity.
	TotalSizeBytes         int64
	MaxSizeBytes           int64
	ConfiguredMaxSizeBytes int64
	NumFiles               int
	// Hot cache (in-memory tier) metrics.
	HotCacheEntries    int
	HotCacheSizeBytes  int64
	HotCacheMaxBytes   int64
	HotCacheHits       int
	HotCacheEvictions  int
	HotCachePromotions int
}

// SaveRate is successful saves over save attempts (saves+errors), 0 when none.
func (s *PagedSSDStats) SaveRate() float64 {
	total := s.Saves + s.Errors
	if total == 0 {
		return 0.0
	}
	return float64(s.Saves) / float64(total)
}

// RecordSave records a successful save.
func (s *PagedSSDStats) RecordSave() { s.Saves++ }

// RecordLoad records a successful load (which is also a hit).
func (s *PagedSSDStats) RecordLoad() {
	s.Loads++
	s.Hits++
}

// RecordError records an error.
func (s *PagedSSDStats) RecordError() { s.Errors++ }

// Reset zeroes the runtime statistics.
func (s *PagedSSDStats) Reset() {
	s.BaseStats.Reset()
	s.Saves = 0
	s.Loads = 0
	s.Errors = 0
	s.SSDWriteDrops = 0
}

// Snapshot renders the SSD stats in the reference key order.
func (s *PagedSSDStats) Snapshot() string {
	pairs := s.baseDict()
	pairs = append(pairs,
		kv{"saves", s.Saves},
		kv{"loads", s.Loads},
		kv{"errors", s.Errors},
		kv{"ssd_write_drops", s.SSDWriteDrops},
		kv{"total_size_bytes", s.TotalSizeBytes},
		kv{"max_size_bytes", s.MaxSizeBytes},
		kv{"configured_max_size_bytes", s.ConfiguredMaxSizeBytes},
		kv{"num_files", s.NumFiles},
		kv{"hot_cache_entries", s.HotCacheEntries},
		kv{"hot_cache_size_bytes", s.HotCacheSizeBytes},
		kv{"hot_cache_max_bytes", s.HotCacheMaxBytes},
		kv{"hot_cache_hits", s.HotCacheHits},
		kv{"hot_cache_evictions", s.HotCacheEvictions},
		kv{"hot_cache_promotions", s.HotCachePromotions},
		kv{"total_queries", s.TotalQueries()},
		kv{"hit_rate", s.HitRate()},
		kv{"save_rate", s.SaveRate()},
	)
	return encodeOrdered(pairs)
}
