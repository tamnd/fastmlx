// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"io"
	"strconv"
	"sync"
)

// sseBufPool recycles encode buffers so the content-delta hot path is alloc-free
// after warmup.
var sseBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 512)
		return &b
	},
}

// ChunkEncoder writes chat-completion SSE events for one request. The id, model,
// and created timestamp are baked into a fixed byte prefix/suffix once, so the
// per-token content delta is a buffer append plus a direct JSON-string escape - // no encoding/json call on the hot path. The role and final chunks (rare) use
// encoding/json for clarity. See spec 1990, 04_http_api_routes.md.
type ChunkEncoder struct {
	id      string
	model   string
	created int64

	deltaPrefix []byte // data: {...,"choices":[{"index":0,"delta":{"content":"
	deltaSuffix []byte // "},"finish_reason":null}]}\n\n
}

// NewChunkEncoder pre-templates the static framing for a request's stream.
func NewChunkEncoder(id, model string, created int64) *ChunkEncoder {
	e := &ChunkEncoder{id: id, model: model, created: created}

	var p []byte
	p = append(p, "data: {\"id\":"...)
	p = appendJSONString(p, id)
	p = append(p, ",\"object\":\"chat.completion.chunk\",\"created\":"...)
	p = strconv.AppendInt(p, created, 10)
	p = append(p, ",\"model\":"...)
	p = appendJSONString(p, model)
	p = append(p, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""...)
	e.deltaPrefix = p

	e.deltaSuffix = []byte("\"},\"finish_reason\":null}]}\n\n")
	return e
}

// WriteContentDelta writes one content-delta SSE event. It is the hot path and
// allocates nothing after the buffer pool warms up.
func (e *ChunkEncoder) WriteContentDelta(w io.Writer, text string) error {
	bufp := sseBufPool.Get().(*[]byte)
	buf := (*bufp)[:0]
	buf = append(buf, e.deltaPrefix...)
	buf = appendEscaped(buf, text)
	buf = append(buf, e.deltaSuffix...)
	_, err := w.Write(buf)
	*bufp = buf
	sseBufPool.Put(bufp)
	return err
}

// WriteRole writes the opening chunk carrying the assistant role.
func (e *ChunkEncoder) WriteRole(w io.Writer) error {
	chunk := ChatCompletionChunk{
		ID: e.id, Object: "chat.completion.chunk", Created: e.created, Model: e.model,
		Choices: []ChunkChoice{{Index: 0, Delta: Delta{Role: "assistant"}}},
	}
	return writeSSEJSON(w, chunk)
}

// WriteFinish writes the terminal chunk carrying finish_reason and (optionally)
// usage.
func (e *ChunkEncoder) WriteFinish(w io.Writer, finishReason string, usage *Usage) error {
	fr := finishReason
	chunk := ChatCompletionChunk{
		ID: e.id, Object: "chat.completion.chunk", Created: e.created, Model: e.model,
		Choices: []ChunkChoice{{Index: 0, Delta: Delta{}, FinishReason: &fr}},
		Usage:   usage,
	}
	return writeSSEJSON(w, chunk)
}

// WriteDone writes the OpenAI stream terminator.
func (e *ChunkEncoder) WriteDone(w io.Writer) error {
	_, err := io.WriteString(w, "data: [DONE]\n\n")
	return err
}

func writeSSEJSON(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	_, err = io.WriteString(w, "\n\n")
	return err
}

// appendJSONString appends s as a quoted, escaped JSON string (with surrounding
// quotes).
func appendJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	dst = appendEscaped(dst, s)
	dst = append(dst, '"')
	return dst
}

const hexDigits = "0123456789abcdef"

// appendEscaped appends s to dst with JSON string escaping, matching
// encoding/json's default (HTML-safe) escaping so every chunk is encoded
// identically regardless of path.
func appendEscaped(dst []byte, s string) []byte {
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < 0x80 {
			if safeByte(b) {
				i++
				continue
			}
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			switch b {
			case '"':
				dst = append(dst, '\\', '"')
			case '\\':
				dst = append(dst, '\\', '\\')
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			default:
				dst = append(dst, '\\', 'u', '0', '0', hexDigits[b>>4], hexDigits[b&0xf])
			}
			i++
			start = i
			continue
		}
		// Multibyte runes pass through unescaped except U+2028/U+2029, which
		// encoding/json escapes for JS safety.
		if i+2 < len(s) && s[i] == 0xe2 && s[i+1] == 0x80 && (s[i+2] == 0xa8 || s[i+2] == 0xa9) {
			if start < i {
				dst = append(dst, s[start:i]...)
			}
			dst = append(dst, '\\', 'u', '2', '0', '2', hexDigits[s[i+2]&0xf])
			i += 3
			start = i
			continue
		}
		i++
	}
	if start < len(s) {
		dst = append(dst, s[start:]...)
	}
	return dst
}

// safeByte reports whether an ASCII byte can be copied verbatim into a JSON
// string. Matches encoding/json: control chars, quote, backslash, and the HTML
// trio <, >, & are escaped.
func safeByte(b byte) bool {
	if b < 0x20 {
		return false
	}
	switch b {
	case '"', '\\', '<', '>', '&':
		return false
	}
	return true
}
