// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"sort"
	"strings"
)

// This file holds the ModelScope recommended and search result cores, the
// counterparts to the Hugging Face builders. They sit on the already-ported
// ParseMSModelEntry normalizer; the ModelScope SDK / REST query that produces
// the raw entries and the per-model enrichment fetch both stay caller seams.
// ModelScope's popularity floor is lower than the hub's.

// MSMinDownloads is the download floor a ModelScope model must clear to appear
// in the recommended lists.
const MSMinDownloads = 50

// msInt reads a normalized entry's numeric field the way x.get(key, 0) does once
// the value has been coerced, returning 0 for a missing or non-numeric value.
func msInt(v any) int {
	n, _ := numToInt(v)
	return n
}

// MSSearchModels filters raw ModelScope entries by a case-insensitive substring
// of the model Name, normalizes the matches, sorts by downloads when that sort
// is requested (created/updated and trending keep the API order), and truncates
// to the limit. It returns the truncated models list and the pre-truncation
// total. The ModelScope query is a caller seam.
func MSSearchModels(rawEntries []map[string]any, query, sortBy string, limit int) map[string]any {
	queryLower := strings.ToLower(query)
	filtered := []map[string]any{}
	for _, entry := range rawEntries {
		name := pyStr(entry["Name"])
		if strings.Contains(strings.ToLower(name), queryLower) {
			filtered = append(filtered, ParseMSModelEntry(entry))
		}
	}
	if sortBy == "downloads" {
		sort.SliceStable(filtered, func(i, j int) bool {
			return msInt(filtered[i]["downloads"]) > msInt(filtered[j]["downloads"])
		})
	}
	total := len(filtered)
	results := filtered
	if len(results) > limit {
		results = results[:limit]
	}
	return map[string]any{"models": results, "total": total}
}

// MSRecommendedPreFilter is the first half of the recommended builder, run
// before the per-model enrichment fetch. It normalizes each raw entry, drops the
// ones below the download floor or already known to exceed memory (a zero size
// is kept since enrichment may reveal it later), and stops once twice the result
// limit is collected so enrichment stays bounded.
func MSRecommendedPreFilter(rawEntries []map[string]any, maxMemoryBytes, resultLimit int) []map[string]any {
	results := []map[string]any{}
	for _, entry := range rawEntries {
		m := ParseMSModelEntry(entry)
		if msInt(m["downloads"]) < MSMinDownloads {
			continue
		}
		size := msInt(m["size"])
		if size > 0 && size > maxMemoryBytes {
			continue
		}
		results = append(results, m)
		if len(results) >= resultLimit*2 {
			break
		}
	}
	return results
}

// MSRecommendedSplit is the second half, run on the enriched entries. It
// re-applies the memory filter now that enrichment may have supplied a real size
// (entries still reporting zero are kept rather than hidden), then forms the
// trending list from the kept order and the popular list by sorting on downloads
// descending, each truncated to the result limit.
func MSRecommendedSplit(enriched []map[string]any, maxMemoryBytes, resultLimit int) map[string]any {
	models := []map[string]any{}
	for _, m := range enriched {
		size := msInt(m["size"])
		if size == 0 || size <= maxMemoryBytes {
			models = append(models, m)
		}
	}
	trending := truncateModels(models, resultLimit)

	popular := make([]map[string]any, len(models))
	copy(popular, models)
	sort.SliceStable(popular, func(i, j int) bool {
		return msInt(popular[i]["downloads"]) > msInt(popular[j]["downloads"])
	})
	popular = truncateModels(popular, resultLimit)

	return map[string]any{"trending": trending, "popular": popular}
}

// truncateModels returns the first n entries, or all of them when there are
// fewer than n.
func truncateModels(models []map[string]any, n int) []map[string]any {
	if len(models) > n {
		return models[:n]
	}
	return models
}
