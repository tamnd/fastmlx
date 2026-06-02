// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"regexp"
	"strings"
)

// Structured-output helpers for response_format. ExtractJSONFromText pulls a
// JSON value out of free-form model output, and BuildJSONSystemPrompt produces
// the prompt instruction that nudges models without native JSON mode to emit
// well-formed output. Both are pure and need no model toolkit.
//
// Schema validation (the json_schema verdict) is intentionally not ported here:
// the reference defers to a JSON Schema validator whose error-message strings
// are library-specific, so it cannot be matched byte for byte without pulling
// in a matching validator. The prompt-injection path, which is what actually
// shapes generation, is fully portable and lands now.

var (
	reCodeBlock = regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)\\s*```")
	reJSONObj   = regexp.MustCompile(`(?s)(\{.*\})`)
	reJSONArr   = regexp.MustCompile(`(?s)(\[.*\])`)
)

// ExtractJSONFromText tries, in order: parse the whole (trimmed) text as JSON,
// parse the contents of any ```json fenced block, then grab the first {...} or
// [...] span and parse that. It returns the parsed value and true on the first
// success, or false when no strategy yields valid JSON. The returned value
// preserves object key order so it re-serializes identically to the reference.
func ExtractJSONFromText(text string) (jval, bool) {
	text = strings.TrimSpace(text)

	if v, ok := parseOrdered(text); ok {
		return v, true
	}

	for _, m := range reCodeBlock.FindAllStringSubmatch(text, -1) {
		if v, ok := parseOrdered(strings.TrimSpace(m[1])); ok {
			return v, true
		}
	}

	for _, re := range []*regexp.Regexp{reJSONObj, reJSONArr} {
		if m := re.FindStringSubmatch(text); m != nil {
			if v, ok := parseOrdered(m[1]); ok {
				return v, true
			}
		}
	}

	return jval{}, false
}

// BuildJSONSystemPrompt returns the system-prompt instruction for a given
// response_format, or "" when none is needed (text format, no format, or an
// unknown type). For json_schema it embeds the schema rendered exactly as
// Python's json.dumps(schema, indent=2): two-space indent, keys in source
// order, a space after each ':'.
func BuildJSONSystemPrompt(rf *ResponseFormat) string {
	if rf == nil {
		return ""
	}

	switch rf.Type {
	case "json_object":
		return "You must respond with valid JSON only. " +
			"Do not include any explanation or text outside the JSON object."

	case "json_schema":
		spec, _ := parseOrdered(string(rf.JSONSchema))

		name := "response"
		if f, ok := spec.getField("name"); ok && f.kind == kindString {
			name = f.s
		}
		desc := ""
		if f, ok := spec.getField("description"); ok && f.kind == kindString {
			desc = f.s
		}
		schema := jval{kind: kindObject}
		if f, ok := spec.getField("schema"); ok {
			schema = f
		}

		var b strings.Builder
		b.WriteString("You must respond with valid JSON matching the '")
		b.WriteString(name)
		b.WriteString("' schema.")
		if desc != "" {
			b.WriteString(" ")
			b.WriteString(desc)
		}
		b.WriteString("\n\nJSON Schema:\n```json\n")
		b.WriteString(schema.dumpIndent())
		b.WriteString("\n```\n\nRespond with only the JSON object, no additional text or explanation.")
		return b.String()

	default: // "text", "", or anything unrecognized
		return ""
	}
}

// dumpIndent renders the value as Python's json.dumps(..., indent=2): non-empty
// containers open on a new line with two-space-per-level indentation, empty
// containers stay on one line ("{}" / "[]"), and scalars use the same form as
// the compact dump.
func (v jval) dumpIndent() string {
	var b strings.Builder
	v.writeIndent(&b, 0)
	return b.String()
}

func (v jval) writeIndent(b *strings.Builder, depth int) {
	switch v.kind {
	case kindObject:
		if len(v.obj) == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteString("{\n")
		for i, kv := range v.obj {
			if i > 0 {
				b.WriteString(",\n")
			}
			b.WriteString(indentSpaces(depth + 1))
			pythonQuote(b, kv.k)
			b.WriteString(": ")
			kv.v.writeIndent(b, depth+1)
		}
		b.WriteByte('\n')
		b.WriteString(indentSpaces(depth))
		b.WriteByte('}')
	case kindArray:
		if len(v.arr) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteString("[\n")
		for i, item := range v.arr {
			if i > 0 {
				b.WriteString(",\n")
			}
			b.WriteString(indentSpaces(depth + 1))
			item.writeIndent(b, depth+1)
		}
		b.WriteByte('\n')
		b.WriteString(indentSpaces(depth))
		b.WriteByte(']')
	default:
		v.write(b)
	}
}

func indentSpaces(depth int) string {
	return strings.Repeat("  ", depth)
}
