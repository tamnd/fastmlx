// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"bytes"
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// This file ports the slice of Python's json.dumps default behaviour that the
// MCP tool-result formatter relies on: separators (", ", ": "), ensure_ascii
// escaping of every non-ASCII rune, and object keys emitted in their original
// order. Go's encoding/json sorts map keys and emits raw UTF-8, so a faithful
// re-serialization of a tool's raw JSON content needs its own encoder.

// pyJSONDumps re-serializes a raw JSON value the way Python's json.dumps does by
// default. It walks the decoder's token stream so object keys keep their source
// order, joins items with ", " and key/value pairs with ": ", and escapes every
// rune outside printable ASCII as a \uXXXX sequence. An empty input is treated
// as the null literal, matching json.dumps(None).
func pyJSONDumps(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "null"
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.UseNumber()
	var b strings.Builder
	if err := pyEncodeValue(dec, &b); err != nil {
		return "null"
	}
	return b.String()
}

func pyEncodeValue(dec *json.Decoder, b *strings.Builder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			b.WriteByte('{')
			first := true
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return err
				}
				if !first {
					b.WriteString(", ")
				}
				first = false
				pyEncodeString(keyTok.(string), b)
				b.WriteString(": ")
				if err := pyEncodeValue(dec, b); err != nil {
					return err
				}
			}
			if _, err := dec.Token(); err != nil { // consume '}'
				return err
			}
			b.WriteByte('}')
		case '[':
			b.WriteByte('[')
			first := true
			for dec.More() {
				if !first {
					b.WriteString(", ")
				}
				first = false
				if err := pyEncodeValue(dec, b); err != nil {
					return err
				}
			}
			if _, err := dec.Token(); err != nil { // consume ']'
				return err
			}
			b.WriteByte(']')
		}
	case string:
		pyEncodeString(t, b)
	case json.Number:
		b.WriteString(pyEncodeNumber(string(t)))
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	}
	return nil
}

// pyEncodeNumber renders a JSON number token as Python's json.dumps would. A
// token with no fractional part or exponent is an int and is emitted verbatim;
// otherwise it is a float, so it round-trips through repr(float) formatting.
func pyEncodeNumber(tok string) string {
	if !strings.ContainsAny(tok, ".eE") {
		return tok
	}
	f, err := strconv.ParseFloat(tok, 64)
	if err != nil {
		return tok
	}
	return formatPyFloat(f)
}

// formatPyFloat reproduces repr(float)/json.dumps float formatting: fixed
// notation for zero or a magnitude in [1e-4, 1e16) and scientific notation
// otherwise, always the shortest decimal that round-trips and always carrying a
// fractional part so it reads back as a float.
func formatPyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}
	abs := math.Abs(f)
	if f == 0 || (abs >= 1e-4 && abs < 1e16) {
		s := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.ContainsRune(s, '.') {
			s += ".0"
		}
		return s
	}
	return strconv.FormatFloat(f, 'e', -1, 64)
}

// pyEncodeString writes a JSON string with ensure_ascii=True escaping: the short
// escapes for the common control characters, \uXXXX for the remaining control
// characters and every non-ASCII rune (a surrogate pair above the basic plane),
// and printable ASCII passed through unchanged.
func pyEncodeString(s string, b *strings.Builder) {
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
			case r >= 0x20 && r < 0x7f:
				b.WriteByte(byte(r))
			case r <= 0xffff:
				writeUEscape(b, r)
			default:
				v := r - 0x10000
				writeUEscape(b, 0xd800+(v>>10))
				writeUEscape(b, 0xdc00+(v&0x3ff))
			}
		}
	}
	b.WriteByte('"')
}

const lowerHex = "0123456789abcdef"

func writeUEscape(b *strings.Builder, r rune) {
	b.WriteString(`\u`)
	b.WriteByte(lowerHex[(r>>12)&0xf])
	b.WriteByte(lowerHex[(r>>8)&0xf])
	b.WriteByte(lowerHex[(r>>4)&0xf])
	b.WriteByte(lowerHex[r&0xf])
}
