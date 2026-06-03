// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"errors"
	"regexp"
	"strconv"
	"strings"
)

// errNoGemma4Call mirrors the reference's ValueError when the fallback parser
// finds no call:name{...} block. The message text matches the reference so the
// caller can surface it unchanged.
var errNoGemma4Call = errors.New("No function call found in Gemma 4 format")

// errGemma4Args reports that a call's argument block could not be coerced into
// JSON even after the robust repairs (the reference would propagate a
// json.JSONDecodeError out of the parser at this point).
var errGemma4Args = errors.New("could not parse Gemma 4 tool-call arguments")

var (
	// gemma4StringDelim captures <|"|>...<|"|> delimited strings (DOTALL,
	// non-greedy) so the inner text can be pulled out before key/value repair.
	gemma4StringDelim = regexp.MustCompile(`(?s)<\|"\|>(.*?)<\|"\|>`)
	// gemma4BareKey quotes a bare key after { or , . The reference uses a
	// lookbehind so the { or , is not consumed; RE2 has no lookbehind, so the
	// boundary char is captured as group 1 and re-emitted in the replacement,
	// which is equivalent for this grammar (each key is preceded by its own
	// brace or comma, never a boundary shared with the previous match).
	gemma4BareKey = regexp.MustCompile(`([{,])\s*(\w+)\s*:`)
	// gemma4BareValue matches a bare (unquoted, non-container) value up to the
	// next , or }. Its first character excludes the quote and the structural
	// punctuation, so an already-quoted value is left untouched.
	gemma4BareValue = regexp.MustCompile(`(:\s*)([^",\[\]{}\s][^,}]*?)(\s*[,}])`)
)

// ParseGemma4ToolCallFallback ports _parse_gemma4_tool_call_fallback
// (api/tool_calling.py:345): a robust fallback for the Gemma 4 call:name{args}
// tool-call format, used only when mlx-lm's own parser fails. It finds every
// call:name{...} block (the name may carry colons, dots, and hyphens; the brace
// block may nest), parses each block's arguments as strict JSON first, and falls
// back to ArgsToJSONRobust for the lenient shapes the model emits. A single call
// returns the object directly; multiple calls return an array, matching the
// reference's "results[0] if len==1 else results".
//
// The reference's recursive brace subpattern (?2) and the lookbehind in the
// key-quoting step are not expressible in RE2, so the call scan uses a depth
// counter for balanced braces and the key rewrite captures the boundary char.
// The toolkit's \w and \s are Unicode; RE2's are ASCII, so a non-ASCII key or
// exotic whitespace would diverge, but the format's keys are ASCII identifiers.
func ParseGemma4ToolCallFallback(text string) (jval, error) {
	calls := findGemma4Calls(text)
	if len(calls) == 0 {
		return jval{}, errNoGemma4Call
	}
	results := make([]jval, 0, len(calls))
	for _, c := range calls {
		args, ok := parseOrdered(c.args)
		if !ok {
			args, ok = ArgsToJSONRobust(c.args)
			if !ok {
				return jval{}, errGemma4Args
			}
		}
		results = append(results, jobj("name", jstr(c.name), "arguments", args))
	}
	if len(results) == 1 {
		return results[0], nil
	}
	return jval{kind: kindArray, arr: results}, nil
}

type gemma4Call struct {
	name string
	args string
}

// findGemma4Calls emulates regex.finditer over call:([\w:.-]+)(\{...\}) with a
// balanced-brace inner group. It scans for each "call:" literal in order; a
// position that does not continue into a name followed by a balanced brace block
// is skipped (advance by one, as the engine would), and a successful match
// resumes scanning after the matched block so nested or trailing calls are
// non-overlapping.
func findGemma4Calls(text string) []gemma4Call {
	var calls []gemma4Call
	i := 0
	for i < len(text) {
		rel := strings.Index(text[i:], "call:")
		if rel < 0 {
			break
		}
		start := i + rel
		p := start + len("call:")
		j := p
		for j < len(text) && isGemma4NameByte(text[j]) {
			j++
		}
		if j == p || j >= len(text) || text[j] != '{' {
			i = start + 1
			continue
		}
		end, ok := matchBalancedBraces(text, j)
		if !ok {
			i = start + 1
			continue
		}
		calls = append(calls, gemma4Call{name: text[p:j], args: text[j:end]})
		i = end
	}
	return calls
}

// isGemma4NameByte reports whether b is allowed in a Gemma 4 function name,
// matching the ASCII reach of the reference's [\w:.-] class.
func isGemma4NameByte(b byte) bool {
	switch {
	case b >= '0' && b <= '9':
		return true
	case b >= 'A' && b <= 'Z':
		return true
	case b >= 'a' && b <= 'z':
		return true
	case b == '_' || b == ':' || b == '.' || b == '-':
		return true
	}
	return false
}

// matchBalancedBraces returns the index just past the brace block that opens at
// start (text[start] must be '{'), or false if the braces never balance. Braces
// are ASCII, so byte scanning is safe over UTF-8 content.
func matchBalancedBraces(text string, start int) (int, bool) {
	depth := 0
	for k := start; k < len(text); k++ {
		switch text[k] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return k + 1, true
			}
		}
	}
	return 0, false
}

// ArgsToJSONRobust ports _gemma4_args_to_json_robust (api/tool_calling.py:296):
// it coerces a Gemma 4 argument blob into a JSON value, handling the shapes
// mlx-lm's parser rejects, namely bare string values without <|"|> delimiters
// and whitespace after commas. The steps mirror the reference: pull out the
// delimited strings behind NUL placeholders, quote bare keys, restore the
// strings as JSON-escaped literals (json.dumps default ensure_ascii=True), and
// try a strict parse; if that fails, quote the remaining bare values (leaving
// numbers, booleans, null, and anything already valid JSON alone) and parse
// again. The second parse can still fail, reported as ok=false.
func ArgsToJSONRobust(argsStr string) (jval, bool) {
	var captured []string
	text := gemma4StringDelim.ReplaceAllStringFunc(argsStr, func(m string) string {
		inner := m[len(`<|"|>`) : len(m)-len(`<|"|>`)]
		captured = append(captured, inner)
		return "\x00" + strconv.Itoa(len(captured)-1) + "\x00"
	})

	text = gemma4BareKey.ReplaceAllString(text, `${1} "${2}":`)

	for i, s := range captured {
		var b strings.Builder
		escapeString(&b, s, true)
		text = strings.ReplaceAll(text, "\x00"+strconv.Itoa(i)+"\x00", b.String())
	}

	if v, ok := parseOrdered(text); ok {
		return v, true
	}

	text = gemma4QuoteBareValues(text)
	return parseOrdered(text)
}

// gemma4QuoteBareValues applies the reference's _quote_bare pass: each bare value
// is stripped, kept verbatim when it is true/false/null (case-insensitive on the
// keyword but the original text is emitted) or already valid JSON, and otherwise
// re-emitted as a JSON-escaped string. The colon-space prefix is normalized and
// the trailing , or } suffix is preserved, matching the reference replacement.
func gemma4QuoteBareValues(text string) string {
	var b strings.Builder
	last := 0
	for _, m := range gemma4BareValue.FindAllStringSubmatchIndex(text, -1) {
		b.WriteString(text[last:m[0]])
		value := strings.TrimSpace(text[m[4]:m[5]])
		suffix := text[m[6]:m[7]]
		lower := strings.ToLower(value)
		if lower == "true" || lower == "false" || lower == "null" {
			b.WriteString(": " + value + suffix)
		} else if _, ok := parseOrdered(value); ok {
			b.WriteString(": " + value + suffix)
		} else {
			var sb strings.Builder
			escapeString(&sb, value, true)
			b.WriteString(": " + sb.String() + suffix)
		}
		last = m[1]
	}
	b.WriteString(text[last:])
	return b.String()
}
