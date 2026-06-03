// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"errors"
	"strings"
)

// CoerceToolCallArguments normalizes a tool_call.arguments value to a JSON-object
// string, porting the reference request-side validator. Native tool-calling chat
// templates iterate the arguments object, so a value that cannot represent a JSON
// object is rejected with an error here (surfacing as a 422) rather than crashing
// template rendering on the next turn.
//
// An object is rendered with json.dumps semantics, normalizing its numbers the
// way a parse-then-dump round trip does (the request body was already decoded, so
// the reference re-emits canonical numbers). A string is accepted only when it is
// empty or whitespace (normalizing to "{}") or parses back to a JSON object, in
// which case the original string is returned verbatim, unstripped and not
// re-serialized. A string that parses to a non-object, a string that fails to
// parse, and any non-string non-object value all return an error.
//
// The reference error messages embed Python type names and the json exception
// text, which do not reproduce in Go, so callers must not depend on the message
// bytes; only the accept/reject decision and the returned string are contractual.
func CoerceToolCallArguments(v jval) (string, error) {
	switch v.kind {
	case kindObject:
		return canonicalizeJSONNumbers(v).dump(), nil
	case kindString:
		stripped := strings.TrimSpace(v.s)
		if stripped == "" {
			return "{}", nil
		}
		parsed, ok := parseOrdered(stripped)
		if !ok {
			return "", errors.New("arguments must be valid JSON")
		}
		if parsed.kind != kindObject {
			return "", errors.New("arguments must be a JSON object")
		}
		return v.s, nil
	default:
		return "", errors.New("arguments must be a JSON-encoded string")
	}
}

// NormalizeFunctionName strips surrounding whitespace from a function name,
// porting the reference field validator. A string value is trimmed; any other
// value passes through unchanged.
func NormalizeFunctionName(v jval) jval {
	if v.kind == kindString {
		return jstr(strings.TrimSpace(v.s))
	}
	return v
}
