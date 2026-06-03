// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"strconv"
	"strings"
)

// SerializeToolCallArguments coerces parser output into a JSON-object arguments
// string, mirroring the reference helper of the same name. Chat templates for
// models with native tool calling iterate the arguments object when a call is
// echoed back in history, so anything that is not a JSON object is coerced to
// "{}" rather than handed back as a non-object the next turn's template would
// choke on.
//
// An object is rendered directly. A string is accepted only when it parses back
// to a JSON object, in which case it is re-rendered (so the output is the
// canonical json.dumps form, not the original spelling). Every other shape,
// including a string that parses to a non-object or fails to parse, yields "{}".
func SerializeToolCallArguments(arguments jval) string {
	switch arguments.kind {
	case kindObject:
		return canonicalizeJSONNumbers(arguments).dump()
	case kindString:
		if parsed, ok := parseOrdered(arguments.s); ok && parsed.kind == kindObject {
			return canonicalizeJSONNumbers(parsed).dump()
		}
	}
	return "{}"
}

// canonicalizeJSONNumbers rewrites every number literal in v to the form
// Python's json.dumps emits after a json.loads round-trip, leaving objects in
// source order and other kinds untouched. This makes a re-rendered string match
// the reference, whose string branch parses then dumps.
func canonicalizeJSONNumbers(v jval) jval {
	switch v.kind {
	case kindObject:
		out := jval{kind: kindObject, obj: make([]jkv, len(v.obj))}
		for i, kv := range v.obj {
			out.obj[i] = jkv{kv.k, canonicalizeJSONNumbers(kv.v)}
		}
		return out
	case kindArray:
		out := jval{kind: kindArray, arr: make([]jval, len(v.arr))}
		for i, item := range v.arr {
			out.arr[i] = canonicalizeJSONNumbers(item)
		}
		return out
	case kindNumber:
		return jval{kind: kindNumber, s: canonicalNumberLiteral(v.s)}
	default:
		return v
	}
}

// canonicalNumberLiteral renders a JSON number literal the way Python's json
// module would after parsing it. A literal carrying a fraction or exponent is a
// float: reparse and render with repr(float) rules. Otherwise it is an integer;
// JSON forbids leading zeros so the literal is already canonical, save for "-0"
// which collapses to "0".
func canonicalNumberLiteral(lit string) string {
	if strings.ContainsAny(lit, ".eE") {
		if f, err := strconv.ParseFloat(lit, 64); err == nil {
			return formatPyFloat(f)
		}
		return lit
	}
	if lit == "-0" {
		return "0"
	}
	return lit
}

// ExtractToolNames collects the function names from OpenAI-format tool
// definitions into a set. Non-object tools, non-object function fields, and
// missing or empty names are skipped, matching the reference.
func ExtractToolNames(tools []jval) map[string]struct{} {
	names := make(map[string]struct{})
	for _, tool := range tools {
		if tool.kind != kindObject {
			continue
		}
		fn, ok := tool.getField("function")
		if !ok || fn.kind != kindObject {
			continue
		}
		if name := fn.getString("name"); name != "" {
			names[name] = struct{}{}
		}
	}
	return names
}
