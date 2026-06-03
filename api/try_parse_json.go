// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// TryParseJSON parses a string as a JSON object or array, returning the parsed
// value when it is valid JSON and the (trimmed) original string otherwise.
// Ported from _try_parse_json in api/utils.py, it exists because the Harmony
// chat template runs a tojson filter over tool-call arguments: a string that is
// already JSON would be double-encoded, so it is decoded to an object first,
// while genuinely non-JSON text is left as a string for the filter to quote.
//
// The string is trimmed first; an empty result, or one that does not begin with
// { or [, is returned as a string without attempting a parse. This is why "42",
// "true", and a bare quoted string stay strings even though they are valid JSON
// scalars: only object and array payloads are unwrapped. A value that begins
// with { or [ but fails to parse as a whole (malformed, or with trailing data)
// also falls back to the trimmed string.
func TryParseJSON(s string) jval {
	s = strings.TrimSpace(s)
	if s == "" {
		return jstr(s)
	}
	if !strings.HasPrefix(s, "{") && !strings.HasPrefix(s, "[") {
		return jstr(s)
	}
	if v, ok := parseOrdered(s); ok {
		return v
	}
	return jstr(s)
}
