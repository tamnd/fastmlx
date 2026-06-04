// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"errors"
	"regexp"
	"strings"
)

// DeepSeek V4 emits tool calls in a DSML grammar: an outer
// <｜DSML｜tool_calls> block wrapping one or more <｜DSML｜invoke> blocks, each
// carrying <｜DSML｜parameter> entries. The outer markers are stripped by the
// tokenizer before the parser runs, so these are exported only for the callers
// that recognize the block, and the parser below sees just the invoke blocks.
const (
	DeepSeekV4ToolCallStart = "<｜DSML｜tool_calls>"
	DeepSeekV4ToolCallEnd   = "</｜DSML｜tool_calls>"
)

// dsmlInvokeRe and dsmlParamRe port _INVOKE_RE and _PARAM_RE. The (?s) flag is
// the reference re.DOTALL, and the non-greedy bodies stop at the first close
// tag. The \s runs match the ASCII spacing the template emits between
// attributes. RE2 has no named groups overhead here, so the captures are
// positional: invoke is (name, body); param is (key, is_str, value).
var (
	dsmlInvokeRe = regexp.MustCompile(`(?s)<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>`)
	dsmlParamRe  = regexp.MustCompile(`(?s)<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>`)
)

// decodeDSMLValue ports _decode_value. A single leading and a single trailing
// newline are trimmed (models pad values before the close tag), but other
// whitespace inside a string value is preserved. A string parameter is returned
// verbatim; a non-string parameter is JSON-decoded so numbers, bools, null,
// arrays, and objects survive.
//
// The reference falls back to Python's ast.literal_eval when JSON decoding
// fails, to catch Python-style literals a model occasionally emits, and to the
// raw string as a last resort. ast.literal_eval cannot be reproduced fully in Go
// without a Python literal evaluator, so the fallback here covers the literals
// the DeepSeek V4 grammar actually produces for a non-string parameter (the
// True/False/None keywords and a single-quoted string) and otherwise returns
// the raw string, the same last resort the reference reaches. Exotic Python
// literals the grammar never emits (tuples, sets, hex or underscored numbers)
// are not evaluated; they take the raw-string path rather than the value
// ast.literal_eval would compute.
func decodeDSMLValue(raw string, isStr bool) jval {
	raw = strings.TrimPrefix(raw, "\n")
	raw = strings.TrimSuffix(raw, "\n")

	if isStr {
		return jstr(raw)
	}
	if v, ok := parseOrdered(raw); ok {
		return v
	}
	switch raw {
	case "True":
		return jval{kind: kindBool, s: "true"}
	case "False":
		return jval{kind: kindBool, s: "false"}
	case "None":
		return jnull()
	}
	if len(raw) >= 2 && raw[0] == '\'' && raw[len(raw)-1] == '\'' && !strings.Contains(raw, `\`) {
		return jstr(raw[1 : len(raw)-1])
	}
	return jstr(raw)
}

// parseSingleInvoke ports _parse_single_invoke: it walks the parameter blocks in
// one invoke body and builds the arguments object, keeping the parameters in the
// order they appear (the reference dict preserves insertion order).
func parseSingleInvoke(name, body string) jval {
	args := jval{kind: kindObject}
	for _, m := range dsmlParamRe.FindAllStringSubmatch(body, -1) {
		key, isStr, value := m[1], m[2] == "true", m[3]
		args.obj = append(args.obj, jkv{key, decodeDSMLValue(value, isStr)})
	}
	return jobj("name", jstr(name), "arguments", args)
}

// ParseDeepSeekV4ToolCall ports parse_tool_call: it parses the body of a
// <｜DSML｜tool_calls> block into a single {name, arguments} object when there is
// exactly one invoke, or an array of them when there are several. An empty body
// (no invoke block) is an error carrying the reference message.
func ParseDeepSeekV4ToolCall(text string) (jval, error) {
	matches := dsmlInvokeRe.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return jval{}, errors.New("No <｜DSML｜invoke> block found in DeepSeek V4 tool-call text")
	}
	parsed := make([]jval, len(matches))
	for i, m := range matches {
		parsed[i] = parseSingleInvoke(m[1], m[2])
	}
	if len(parsed) == 1 {
		return parsed[0], nil
	}
	return jval{kind: kindArray, arr: parsed}, nil
}
