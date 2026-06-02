// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"math"
	"strconv"
	"strings"
)

// kv is one key/value pair in an ordered JSON object. The cache stat snapshots
// are built as ordered slices so the serialized object keeps the reference key
// order rather than Go's randomized map iteration.
type kv struct {
	k string
	v any
}

// encodeOrdered serializes a slice of key/value pairs as a JSON object, keys in
// slice order. Values may be int, int64, float64, string, or a nested []kv. The
// nesting covers the rate-tracker output, which is a map of objects.
func encodeOrdered(pairs []kv) string {
	var b strings.Builder
	b.WriteByte('{')
	for i, p := range pairs {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Quote(p.k))
		b.WriteByte(':')
		writeValue(&b, p.v)
	}
	b.WriteByte('}')
	return b.String()
}

func writeValue(b *strings.Builder, v any) {
	switch x := v.(type) {
	case int:
		b.WriteString(strconv.Itoa(x))
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case float64:
		b.WriteString(formatFloat(x))
	case string:
		b.WriteString(strconv.Quote(x))
	case []kv:
		b.WriteString(encodeOrdered(x))
	default:
		b.WriteString("null")
	}
}

// formatFloat renders a float as a shortest round-trip decimal. Integer-valued
// floats keep a trailing ".0" so the value reads as a float, matching how the
// ratios are reported.
func formatFloat(f float64) string {
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return "0.0"
	}
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// round4 rounds to four decimals with ties going to the nearest even digit,
// matching Python's round(x, 4). round2 does the same to two decimals.
func round4(x float64) float64 { return roundN(x, 4) }
func round2(x float64) float64 { return roundN(x, 2) }

func roundN(x float64, n int) float64 {
	pow := math.Pow(10, float64(n))
	return math.RoundToEven(x*pow) / pow
}
