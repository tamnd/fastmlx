// SPDX-License-Identifier: MIT OR Apache-2.0

package download

import (
	"sort"
	"strings"
)

// MinDownloads is the popularity floor a model must clear to appear in the
// recommended lists.
const MinDownloads = 100

// HubModel is the slice of a hub list_models entry the recommended and search
// builders read. Parameters is the safetensors per-dtype parameter-count map
// (the safetensors["parameters"] sub-dict); an empty map stands for a model
// without usable safetensors metadata. The hub query that produces these stays a
// caller seam.
type HubModel struct {
	ID            string
	Downloads     int64
	Likes         int64
	TrendingScore int64
	Parameters    map[string]int64
}

// lastSegment returns the text after the final slash, matching id.split("/")[-1].
func lastSegment(id string) string {
	if i := strings.LastIndex(id, "/"); i >= 0 {
		return id[i+1:]
	}
	return id
}

// RecommendedList filters a fetched category (trending or popular) down to the
// models that fit in memory and shapes each into the panel entry, truncating to
// resultLimit. A model is dropped when it has no safetensors parameters, has
// fewer than MinDownloads downloads, or has a computed size that is zero or
// larger than maxMemoryBytes. The entry's params fields are null when the count
// is not positive.
func RecommendedList(models []HubModel, maxMemoryBytes int64, resultLimit int) []map[string]any {
	results := []map[string]any{}
	for _, m := range models {
		if len(m.Parameters) == 0 {
			continue
		}
		if m.Downloads < MinDownloads {
			continue
		}
		size := SafetensorsDiskSize(m.Parameters)
		if size <= 0 || size > maxMemoryBytes {
			continue
		}
		params := ParamCount(m.Parameters)
		var paramsVal, paramsFmt any
		if params > 0 {
			paramsVal = params
			paramsFmt = FormatParamCount(params)
		}
		results = append(results, map[string]any{
			"repo_id":          m.ID,
			"name":             lastSegment(m.ID),
			"downloads":        m.Downloads,
			"likes":            m.Likes,
			"trending_score":   m.TrendingScore,
			"size":             size,
			"size_formatted":   FormatModelSize(size),
			"params":           paramsVal,
			"params_formatted": paramsFmt,
		})
	}
	if len(results) > resultLimit {
		results = results[:resultLimit]
	}
	return results
}

// SearchOptions carries the search query's filters and sort. A nil filter
// pointer means the filter is not applied. Sort is the UI sort option; SortBySize
// forces a size sort regardless of Sort, and SortAscending controls direction
// for the forced size sort.
type SearchOptions struct {
	Sort          string
	Limit         int
	MinParams     *int64
	MaxParams     *int64
	MinSize       *int64
	MaxSize       *int64
	SortBySize    bool
	SortAscending bool
}

// searchRow keeps an entry alongside the keys its sort reads, so the comparison
// matches the Python "params or 0" and size-with-unknown-last behavior.
type searchRow struct {
	entry     map[string]any
	paramsKey int64
	sizeKey   int64
}

// SearchModels builds a result entry for every fetched model, applies the
// parameter- and size-range filters, sorts client-side, and truncates to the
// limit. Unlike the recommended builder it keeps models without parameters
// (their params fields stay null and size reads as zero); the parameter filters
// then drop those rows. Size sorts place unknown-size (zero) entries last by
// keying them at -1. The default ordering keeps the hub's order. The returned
// map carries the truncated models list and the pre-truncation total.
func SearchModels(models []HubModel, opts SearchOptions) map[string]any {
	rows := []searchRow{}
	for _, m := range models {
		var paramsVal, paramsFmt any
		var paramsKey, size int64
		haveParams := false
		if len(m.Parameters) > 0 {
			p := ParamCount(m.Parameters)
			if p > 0 {
				paramsFmt = FormatParamCount(p)
			}
			size = SafetensorsDiskSize(m.Parameters)
			// Mirror "params = count" then the negative-count guard: counts are
			// never negative, so the count (even zero) is kept.
			paramsVal = p
			paramsKey = p
			haveParams = true
		}
		if opts.MinParams != nil && (!haveParams || paramsKey < *opts.MinParams) {
			continue
		}
		if opts.MaxParams != nil && (!haveParams || paramsKey > *opts.MaxParams) {
			continue
		}
		if opts.MinSize != nil && size < *opts.MinSize {
			continue
		}
		if opts.MaxSize != nil && size > *opts.MaxSize {
			continue
		}
		sizeFmt := ""
		if size > 0 {
			sizeFmt = FormatModelSize(size)
		}
		sizeKey := size
		if size <= 0 {
			sizeKey = -1
		}
		rows = append(rows, searchRow{
			entry: map[string]any{
				"repo_id":          m.ID,
				"name":             m.ID,
				"downloads":        m.Downloads,
				"likes":            m.Likes,
				"trending_score":   m.TrendingScore,
				"size":             size,
				"size_formatted":   sizeFmt,
				"params":           paramsVal,
				"params_formatted": paramsFmt,
			},
			paramsKey: paramsKey,
			sizeKey:   sizeKey,
		})
	}

	switch {
	case opts.Sort == "most_params":
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].paramsKey > rows[j].paramsKey })
	case opts.Sort == "least_params":
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].paramsKey < rows[j].paramsKey })
	case opts.Sort == "largest" || opts.Sort == "smallest" || opts.SortBySize:
		reverse := opts.Sort == "largest" || (opts.SortBySize && !opts.SortAscending)
		sort.SliceStable(rows, func(i, j int) bool {
			if reverse {
				return rows[i].sizeKey > rows[j].sizeKey
			}
			return rows[i].sizeKey < rows[j].sizeKey
		})
	}

	total := len(rows)
	entries := make([]map[string]any, len(rows))
	for i, r := range rows {
		entries[i] = r.entry
	}
	if len(entries) > opts.Limit {
		entries = entries[:opts.Limit]
	}
	return map[string]any{"models": entries, "total": total}
}
