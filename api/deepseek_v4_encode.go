// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// EncodeArgumentsToDSML ports encode_arguments_to_dsml: it renders a tool call's
// arguments object as the DSML parameter block DeepSeek V4 emits, one
// <｜DSML｜parameter> line per key, joined by newlines. It is the inverse of the
// DSML parser (ParseDeepSeekV4ToolCall).
//
// The reference accepts the OpenAI shape (arguments as a JSON string) or the
// Anthropic adapter's shape (arguments already a dict) and resolves both to a
// dict before iterating; that coercion is the caller's seam, so this takes the
// already-parsed arguments object (parse a JSON-string form with parseOrdered
// first). A non-object input renders to an empty string, since there are no keys
// to walk.
//
// Per key the value branch mirrors the reference exactly: a string parameter
// carries string="true" and is written raw, with no quoting or escaping (the
// template inserts it verbatim); every other type carries string="false" and is
// serialized with to_json, i.e. json.dumps(ensure_ascii=False). The reference
// reaches to_json on values that have already passed through json.loads (or
// arrived as native Python numbers from the adapter), so number literals are in
// their canonical round-tripped form — canonicalizeJSONNumbers reproduces that
// before dump(), matching how a literal like 1e10 becomes 10000000000.0.
func EncodeArgumentsToDSML(arguments jval) string {
	if arguments.kind != kindObject {
		return ""
	}
	var b strings.Builder
	for i, kv := range arguments.obj {
		if i > 0 {
			b.WriteByte('\n')
		}
		isStr := "false"
		value := canonicalizeJSONNumbers(kv.v).dump()
		if kv.v.kind == kindString {
			isStr = "true"
			value = kv.v.s
		}
		b.WriteString(`<｜DSML｜parameter name="`)
		b.WriteString(kv.k)
		b.WriteString(`" string="`)
		b.WriteString(isStr)
		b.WriteString(`">`)
		b.WriteString(value)
		b.WriteString(`</｜DSML｜parameter>`)
	}
	return b.String()
}
