// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"regexp"
	"strconv"
	"strings"
)

// This file holds two pure display cores: parsing a "Supported models:" bullet
// list out of a structural-tag function's docstring (the docstring lookup is a
// caller seam), and the detailed size formatter the system-info panel uses,
// which carries two decimals at GB and TB. This is a third size variant,
// distinct from FormatSize (one-decimal GB, bytes tier) and FormatModelSize (no
// bytes tier, one-decimal GB).

// supportedModelsDocRe matches a "Supported models:" header followed by one or
// more bullet lines, capturing the bullet block.
var supportedModelsDocRe = regexp.MustCompile(`Supported models:\s*\n((?:\s*-\s*\S.*\n?)+)`)

// ModelsFromDocstring extracts the supported-model names from a docstring's
// "Supported models:" bullet list, returning an empty slice when the section is
// absent. Each kept line is trimmed, has its leading dashes stripped, and is
// trimmed again.
func ModelsFromDocstring(doc string) []string {
	m := supportedModelsDocRe.FindStringSubmatch(doc)
	if m == nil {
		return []string{}
	}
	out := []string{}
	for line := range strings.SplitSeq(m[1], "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "-") {
			out = append(out, strings.TrimSpace(strings.TrimLeft(s, "-")))
		}
	}
	return out
}

// FormatSizeDetailed formats a byte count for the system-info panel: a bytes
// tier below 1 KB, KB and MB with one decimal, and GB and TB with two decimals.
func FormatSizeDetailed(sizeBytes int) string {
	const (
		kb = 1024
		mb = 1024 * 1024
		gb = 1024 * 1024 * 1024
		tb = 1024 * 1024 * 1024 * 1024
	)
	switch {
	case sizeBytes < kb:
		return strconv.Itoa(sizeBytes) + " B"
	case sizeBytes < mb:
		return strconv.FormatFloat(float64(sizeBytes)/kb, 'f', 1, 64) + " KB"
	case sizeBytes < gb:
		return strconv.FormatFloat(float64(sizeBytes)/mb, 'f', 1, 64) + " MB"
	case sizeBytes < tb:
		return strconv.FormatFloat(float64(sizeBytes)/gb, 'f', 2, 64) + " GB"
	default:
		return strconv.FormatFloat(float64(sizeBytes)/tb, 'f', 2, 64) + " TB"
	}
}
