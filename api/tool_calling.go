// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"regexp"
	"strings"
)

// Tool-call extraction turns a model's free text into structured calls. Models
// that emit tool calls do so in a handful of wire formats; this file ports the
// generic, format-specific fallback parsers that need no model toolkit:
//
//   - XML: <tool_call>{json}</tool_call>, <tool_call><function=name>...,
//     and the GLM <arg_key>/<arg_value> shape.
//   - Namespaced: <ns:tool_call><invoke name="..."><parameter .../></invoke>.
//   - Bracket: [Calling tool: name({...})] and [Tool call: name].
//
// The per-family parsers driven by a model's own tokenizer live in the model
// toolkit and are handled by the compute backend, not here.
//
// Argument strings are emitted to match the reference byte for byte: the
// reference re-serializes arguments with Python's json.dumps(ensure_ascii=
// False), which keeps object keys in insertion order and puts a space after
// every ':' and ','. Go's encoding/json does neither, so this file carries an
// order-preserving JSON model (jval) with a Python-compatible serializer.

var (
	reXMLToolCall  = regexp.MustCompile(`(?s)<tool_call>(.*?)</tool_call>`)
	reXMLFunction  = regexp.MustCompile(`(?s)^<function=(\w+)>(.*?)</function>`)
	reXMLParameter = regexp.MustCompile(`(?s)<parameter=(\w+)>\s*(.*?)\s*</parameter>`)
	reArgKey       = regexp.MustCompile(`<arg_key>(.*?)</arg_key>`)
	reArgValue     = regexp.MustCompile(`(?s)<arg_value>(.*?)</arg_value>`)
	reGLMName      = regexp.MustCompile(`(?s)^(.*?)<arg_key>`)

	reInvoke      = regexp.MustCompile(`(?s)<invoke\s+name="([^"]+)">(.*?)</invoke>`)
	reInvokeParam = regexp.MustCompile(`(?s)<parameter\s+name="([^"]+)">(.*?)</parameter>`)

	reBracketWithArgs = regexp.MustCompile(`(?s)\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.\-]*)\((\{.*?\})\)\]`)
	reBracketNoArgs   = regexp.MustCompile(`\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.\-]*)\]`)
)

// newCallID mints an OpenAI-style "call_<hex>" identifier. The reference uses a
// random uuid suffix, so the id is never part of parity comparisons.
func newCallID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}

func makeCall(name, arguments string) ToolCall {
	return ToolCall{
		ID:       newCallID(),
		Type:     "function",
		Function: FunctionCall{Name: name, Arguments: arguments},
	}
}

// ParseXMLToolCalls handles <tool_call>...</tool_call> blocks in three shapes:
// a JSON object, a <function=name> body, or the GLM arg_key/arg_value pairs. It
// returns the text with the tool-call tags removed (trimmed) and the parsed
// calls, or (text, nil) when nothing matched.
func ParseXMLToolCalls(text string) (string, []ToolCall) {
	var calls []ToolCall
	for _, m := range reXMLToolCall.FindAllStringSubmatch(text, -1) {
		content := strings.TrimSpace(m[1])

		// JSON object: {"name": "...", "arguments": {...}}.
		if v, ok := parseOrdered(content); ok && v.kind == kindObject {
			name := v.getString("name")
			arg, present := v.getField("arguments")
			calls = append(calls, makeCall(name, serializeArgs(arg, present)))
			continue
		}

		// <function=name><parameter=key>value</parameter>...</function>.
		if fm := reXMLFunction.FindStringSubmatch(content); fm != nil {
			obj := jval{kind: kindObject}
			for _, pm := range reXMLParameter.FindAllStringSubmatch(fm[2], -1) {
				obj.obj = append(obj.obj, jkv{pm[1], argValue(strings.TrimSpace(pm[2]))})
			}
			calls = append(calls, makeCall(fm[1], obj.dump()))
			continue
		}

		// GLM: name<arg_key>k</arg_key><arg_value>v</arg_value>...
		keys := reArgKey.FindAllStringSubmatch(content, -1)
		if len(keys) > 0 {
			vals := reArgValue.FindAllStringSubmatch(content, -1)
			name := ""
			if nm := reGLMName.FindStringSubmatch(content); nm != nil {
				name = strings.TrimSpace(nm[1])
			} else {
				name = strings.TrimSpace(strings.Split(content, "<")[0])
			}
			obj := jval{kind: kindObject}
			n := min(len(keys), len(vals))
			for i := range n {
				obj.obj = append(obj.obj, jkv{keys[i][1], argValue(vals[i][1])})
			}
			calls = append(calls, makeCall(name, obj.dump()))
		}
	}

	if len(calls) == 0 {
		return text, nil
	}
	cleaned := strings.TrimSpace(reXMLToolCall.ReplaceAllString(text, ""))
	return cleaned, calls
}

// ParseNamespacedToolCalls handles <ns:tool_call><invoke name="...">...</invoke>
// blocks (MiniMax and similar). One block may carry several <invoke>s.
func ParseNamespacedToolCalls(text, namespace string) (string, []ToolCall) {
	pat := regexp.MustCompile(`(?s)` + regexp.QuoteMeta("<"+namespace+":tool_call>") +
		`(.*?)` + regexp.QuoteMeta("</"+namespace+":tool_call>"))

	var calls []ToolCall
	for _, m := range pat.FindAllStringSubmatch(text, -1) {
		content := strings.TrimSpace(m[1])
		for _, im := range reInvoke.FindAllStringSubmatch(content, -1) {
			obj := jval{kind: kindObject}
			for _, pm := range reInvokeParam.FindAllStringSubmatch(im[2], -1) {
				obj.obj = append(obj.obj, jkv{pm[1], argValue(strings.TrimSpace(pm[2]))})
			}
			calls = append(calls, makeCall(im[1], obj.dump()))
		}
	}

	if len(calls) == 0 {
		return text, nil
	}
	cleaned := strings.TrimSpace(pat.ReplaceAllString(text, ""))
	return cleaned, calls
}

// ParseBracketToolCalls handles [Calling tool: name({...})] and the args-less
// [Tool call: name] form. The with-args matches win where the two overlap.
func ParseBracketToolCalls(text string) (string, []ToolCall) {
	type span struct{ start, end int }
	var spans []span
	var calls []ToolCall

	for _, loc := range reBracketWithArgs.FindAllStringSubmatchIndex(text, -1) {
		name := text[loc[2]:loc[3]]
		argsStr := text[loc[4]:loc[5]]
		var args jval
		if v, ok := parseOrdered(argsStr); ok {
			args = v
		} else {
			args = jval{kind: kindObject, obj: []jkv{{"raw", jval{kind: kindString, s: argsStr}}}}
		}
		calls = append(calls, makeCall(name, args.dump()))
		spans = append(spans, span{loc[0], loc[1]})
	}

	for _, loc := range reBracketNoArgs.FindAllStringSubmatchIndex(text, -1) {
		start := loc[0]
		overlap := false
		for _, sp := range spans {
			if sp.start <= start && start < sp.end {
				overlap = true
				break
			}
		}
		if overlap {
			continue
		}
		calls = append(calls, makeCall(text[loc[2]:loc[3]], "{}"))
		spans = append(spans, span{start, loc[1]})
	}

	if len(calls) == 0 {
		return text, nil
	}
	cleaned := reBracketWithArgs.ReplaceAllString(text, "")
	cleaned = strings.TrimSpace(reBracketNoArgs.ReplaceAllString(cleaned, ""))
	return cleaned, calls
}

// serializeArgs mirrors the reference _serialize_tool_call_arguments: a JSON
// object is re-serialized; a string that parses to an object is re-serialized;
// anything else (including a missing arguments field) becomes "{}".
func serializeArgs(arg jval, present bool) string {
	if !present {
		return "{}"
	}
	switch arg.kind {
	case kindObject:
		return arg.dump()
	case kindString:
		if v, ok := parseOrdered(arg.s); ok && v.kind == kindObject {
			return v.dump()
		}
	}
	return "{}"
}

// argValue parses a raw parameter value as JSON, falling back to the literal
// string when it is not valid JSON. This is the reference's
// `try: json.loads(v) except: v` applied to every parameter.
func argValue(raw string) jval {
	if v, ok := parseOrdered(raw); ok {
		return v
	}
	return jval{kind: kindString, s: raw}
}

// --- order-preserving JSON value model ---

const (
	kindObject = 'o'
	kindArray  = 'a'
	kindString = 's'
	kindNumber = 'n'
	kindBool   = 'b'
	kindNull   = 'z'
)

type jval struct {
	kind byte
	s    string // decoded string, number literal, or "true"/"false"
	obj  []jkv  // object members, in source order
	arr  []jval // array elements
}

type jkv struct {
	k string
	v jval
}

func (v jval) getField(key string) (jval, bool) {
	for _, kv := range v.obj {
		if kv.k == key {
			return kv.v, true
		}
	}
	return jval{}, false
}

func (v jval) getString(key string) string {
	if f, ok := v.getField(key); ok && f.kind == kindString {
		return f.s
	}
	return ""
}

// parseOrdered decodes a JSON text into an order-preserving value. It reports
// false on any decode error or trailing content, matching the reference's
// json.loads (which rejects trailing garbage but allows surrounding space).
func parseOrdered(raw string) (jval, bool) {
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return jval{}, false
	}
	if _, err := dec.Token(); err != io.EOF {
		return jval{}, false
	}
	return v, true
}

func decodeValue(dec *json.Decoder) (jval, error) {
	t, err := dec.Token()
	if err != nil {
		return jval{}, err
	}
	switch tok := t.(type) {
	case json.Delim:
		switch tok {
		case '{':
			o := jval{kind: kindObject}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return jval{}, err
				}
				key := kt.(string)
				val, err := decodeValue(dec)
				if err != nil {
					return jval{}, err
				}
				o.obj = append(o.obj, jkv{key, val})
			}
			if _, err := dec.Token(); err != nil { // closing }
				return jval{}, err
			}
			return o, nil
		case '[':
			a := jval{kind: kindArray}
			for dec.More() {
				val, err := decodeValue(dec)
				if err != nil {
					return jval{}, err
				}
				a.arr = append(a.arr, val)
			}
			if _, err := dec.Token(); err != nil { // closing ]
				return jval{}, err
			}
			return a, nil
		}
		return jval{}, io.ErrUnexpectedEOF
	case string:
		return jval{kind: kindString, s: tok}, nil
	case json.Number:
		return jval{kind: kindNumber, s: tok.String()}, nil
	case bool:
		if tok {
			return jval{kind: kindBool, s: "true"}, nil
		}
		return jval{kind: kindBool, s: "false"}, nil
	case nil:
		return jval{kind: kindNull}, nil
	}
	return jval{}, io.ErrUnexpectedEOF
}

// dump renders the value exactly as Python's json.dumps(ensure_ascii=False):
// keys in source order, a space after ':' and ',', non-ASCII left unescaped.
func (v jval) dump() string {
	var b strings.Builder
	v.write(&b)
	return b.String()
}

func (v jval) write(b *strings.Builder) {
	switch v.kind {
	case kindObject:
		b.WriteByte('{')
		for i, kv := range v.obj {
			if i > 0 {
				b.WriteString(", ")
			}
			pythonQuote(b, kv.k)
			b.WriteString(": ")
			kv.v.write(b)
		}
		b.WriteByte('}')
	case kindArray:
		b.WriteByte('[')
		for i, item := range v.arr {
			if i > 0 {
				b.WriteString(", ")
			}
			item.write(b)
		}
		b.WriteByte(']')
	case kindString:
		pythonQuote(b, v.s)
	case kindNumber, kindBool:
		b.WriteString(v.s)
	case kindNull:
		b.WriteString("null")
	}
}

// pythonQuote writes s as a JSON string with ensure_ascii=False (non-ASCII
// passed through), the mode the tool-call re-serialization uses.
func pythonQuote(b *strings.Builder, s string) {
	escapeString(b, s, false)
}

// escapeString writes s as a JSON string using Python's escaping rules: escape
// the quote, backslash, and the named control characters, and emit any other
// control character as \u00xx. When asciiOnly is set (Python's default
// ensure_ascii=True), every character outside printable ASCII is escaped too,
// with a surrogate pair for code points above U+FFFF; otherwise non-ASCII is
// passed through unchanged.
func escapeString(b *strings.Builder, s string, asciiOnly bool) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20:
				writeUnicodeEscape(b, uint32(r))
			case asciiOnly && r > 0x7e:
				if r > 0xffff {
					c := uint32(r) - 0x10000
					writeUnicodeEscape(b, 0xd800+(c>>10))
					writeUnicodeEscape(b, 0xdc00+(c&0x3ff))
				} else {
					writeUnicodeEscape(b, uint32(r))
				}
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

func writeUnicodeEscape(b *strings.Builder, c uint32) {
	b.WriteString(`\u`)
	b.WriteByte(hexDigits[(c>>12)&0xf])
	b.WriteByte(hexDigits[(c>>8)&0xf])
	b.WriteByte(hexDigits[(c>>4)&0xf])
	b.WriteByte(hexDigits[c&0xf])
}
