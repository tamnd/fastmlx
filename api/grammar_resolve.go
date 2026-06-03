// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "strings"

// Request-side resolution of the structured-output grammar shape. These are the
// pure transforms that decide which grammar a request asks for; the step that
// compiles the resulting element into a logit mask lives behind the GPU seam.

// NormalizeStructuredOutputs folds the vLLM-compatible guided_grammar alias into
// the structured_outputs shape. An explicit structured_outputs value (including
// an empty object) is returned unchanged; otherwise a non-empty guided_grammar
// becomes {"grammar": <value>}, and a missing or empty grammar yields null.
// Callers pass jnull() for an absent structured_outputs.
func NormalizeStructuredOutputs(structuredOutputs jval, guidedGrammar string) jval {
	if structuredOutputs.kind != kindNull {
		return structuredOutputs
	}
	if guidedGrammar != "" {
		return jobj("grammar", jstr(guidedGrammar))
	}
	return jnull()
}

// BuildFormatElement turns a request's structured_outputs and response_format
// into a single format element, or null when neither asks for a grammar.
// structured_outputs wins over response_format, and within it exactly one of
// json_schema (alias "json"), grammar, regex, or choice is honored in that
// order. A json_schema given as a string is parsed so the schema embeds as an
// object; a choice list compiles to an EBNF "root ::= " alternation where each
// option is rendered with json.dumps (ensure_ascii, so non-ASCII is \u-escaped).
// For response_format, "json_schema" lifts the nested .json_schema.schema and
// "json_object" produces an empty schema.
func BuildFormatElement(structuredOutputs jval, responseFormat jval) jval {
	if structuredOutputs.kind == kindObject {
		so := structuredOutputs
		if js, ok := firstNonNullField(so, "json_schema", "json"); ok {
			schema := js
			if js.kind == kindString {
				if parsed, pok := parseOrdered(js.s); pok {
					schema = parsed
				}
			}
			return jobj("type", jstr("json_schema"), "json_schema", schema)
		}
		if g, ok := nonNullField(so, "grammar"); ok {
			return jobj("type", jstr("grammar"), "grammar", g)
		}
		if r, ok := nonNullField(so, "regex"); ok {
			return jobj("type", jstr("regex"), "pattern", r)
		}
		if c, ok := nonNullField(so, "choice"); ok {
			parts := make([]string, len(c.arr))
			for i, el := range c.arr {
				parts[i] = el.dumpASCII()
			}
			ebnf := "root ::= " + strings.Join(parts, " | ")
			return jobj("type", jstr("grammar"), "grammar", jstr(ebnf))
		}
	}
	if responseFormat.kind == kindObject {
		rf := responseFormat
		switch rf.getString("type") {
		case "json_schema":
			if jsf, ok := rf.getField("json_schema"); ok && jsf.kind == kindObject {
				if schema, ok2 := nonNullField(jsf, "schema"); ok2 {
					return jobj("type", jstr("json_schema"), "json_schema", schema)
				}
			}
		case "json_object":
			return jobj("type", jstr("json_schema"), "json_schema", jobj())
		}
	}
	return jnull()
}

// SettingsGuidedGrammar returns a model-level guided grammar only when the
// feature is enabled and the grammar is non-empty after trimming, mirroring the
// reference where a disabled flag or a blank grammar both fall back to none ("").
func SettingsGuidedGrammar(enabled bool, grammar string) string {
	if !enabled {
		return ""
	}
	return strings.TrimSpace(grammar)
}

// EffectiveGuidedGrammar chooses the grammar a request effectively uses: the
// request's own guided_grammar alias wins; the model default applies only when
// the request names neither structured_outputs nor response_format; otherwise no
// guided grammar is used (""). The present flags model the reference's
// "is None" checks on those two request fields.
func EffectiveGuidedGrammar(structuredOutputsPresent, responseFormatPresent bool, requestGuidedGrammar, settingsGuidedGrammar string) string {
	if requestGuidedGrammar != "" {
		return requestGuidedGrammar
	}
	if !structuredOutputsPresent && !responseFormatPresent {
		return settingsGuidedGrammar
	}
	return ""
}

// nonNullField returns a field only when it is present and not JSON null, the
// shape the reference's "x.get(k) is not None" checks expect.
func nonNullField(v jval, key string) (jval, bool) {
	f, ok := v.getField(key)
	if !ok || f.kind == kindNull {
		return jval{}, false
	}
	return f, true
}

// firstNonNullField returns the first present, non-null field among keys, used
// to resolve a field that carries an alias.
func firstNonNullField(v jval, keys ...string) (jval, bool) {
	for _, k := range keys {
		if f, ok := nonNullField(v, k); ok {
			return f, true
		}
	}
	return jval{}, false
}
