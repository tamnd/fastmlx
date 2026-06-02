// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"
)

// Fixtures in testdata/parity/anthropic_request.json are captured from the
// reference convert_anthropic_to_internal across the fallback-markup and native
// tool-calling paths, system extraction, image handling, and role merging. The
// internal message list is consumed by the chat-template layer, where object
// key order is not contractual, so the comparison is structural (canonicalized
// JSON) rather than byte-exact. Document blocks are excluded here and covered by
// TestDecodeDocumentBlock, since the reference placeholder names the project and
// uses an em-dash that this port rewords.

type requestFixtures struct {
	Cases []struct {
		Label    string          `json:"label"`
		System   json.RawMessage `json:"system"`
		Messages json.RawMessage `json:"messages"`
		Opts     struct {
			NativeToolCalling      bool `json:"native_tool_calling"`
			PreserveImages         bool `json:"preserve_images"`
			NativeReasoningContent bool `json:"native_reasoning_content"`
		} `json:"opts"`
		Result json.RawMessage `json:"result"`
	} `json:"cases"`
}

func loadRequestFixtures(t testing.TB) requestFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/anthropic_request.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx requestFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("no fixture cases")
	}
	return fx
}

func TestConvertAnthropicToInternalParity(t *testing.T) {
	for _, c := range loadRequestFixtures(t).Cases {
		t.Run(c.Label, func(t *testing.T) {
			system, _ := parseOrdered(string(c.System))

			msgsVal, ok := parseOrdered(string(c.Messages))
			if !ok || msgsVal.kind != kindArray {
				t.Fatalf("messages did not parse as array: %s", c.Messages)
			}
			messages := make([]AnthropicInMessage, 0, len(msgsVal.arr))
			for _, m := range msgsVal.arr {
				content, _ := m.getField("content")
				messages = append(messages, AnthropicInMessage{
					Role:    m.getString("role"),
					Content: content,
				})
			}

			opts := AnthropicConvertOptions{
				NativeToolCalling:      c.Opts.NativeToolCalling,
				PreserveImages:         c.Opts.PreserveImages,
				NativeReasoningContent: c.Opts.NativeReasoningContent,
			}
			result := convertAnthropicToInternal(system, messages, opts)

			got := canonJSON(t, []byte(jval{kind: kindArray, arr: result}.dumpASCII()))
			want := canonJSON(t, c.Result)
			if got != want {
				t.Errorf("case %q\n got  %s\n want %s", c.Label, got, want)
			}
		})
	}
}

func TestDecodeDocumentBlock(t *testing.T) {
	mk := func(mediaType, data, title string) jval {
		src := jobj("type", jstr("base64"), "media_type", jstr(mediaType), "data", jstr(data))
		b := jobj("type", jstr("document"), "source", src)
		if title != "" {
			b = b.setField("title", jstr(title))
		}
		return b
	}
	plain := base64.StdEncoding.EncodeToString([]byte("the body text"))

	tests := []struct {
		name  string
		block jval
		want  string
	}{
		{
			name:  "text_plain_with_title",
			block: mk("text/plain", plain, "Notes"),
			want:  "[Document: Notes]\nthe body text",
		},
		{
			name:  "text_plain_no_title",
			block: mk("text/plain", plain, ""),
			want:  "the body text",
		},
		{
			name:  "text_plain_bad_base64",
			block: mk("text/plain", "!!!not base64!!!", "Bad"),
			want:  "[Document: Bad: failed to decode]",
		},
		{
			name:  "pdf_with_title",
			block: mk("application/pdf", "JVBERi0=", "Report"),
			want:  "[Document: Report (application/pdf): document parsing is not available, send as text instead.]",
		},
		{
			name:  "pdf_no_title",
			block: mk("application/pdf", "JVBERi0=", ""),
			want:  "[Document: untitled (application/pdf): document parsing is not available, send as text instead.]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := decodeDocumentBlock(tt.block); got != tt.want {
				t.Errorf("decodeDocumentBlock\n got  %q\n want %q", got, tt.want)
			}
		})
	}
}

func BenchmarkConvertAnthropicToInternal(b *testing.B) {
	system := jstr("you are helpful")
	content := jval{kind: kindArray, arr: []jval{
		jobj("type", jstr("text"), "text", jstr("let me check")),
		jobj("type", jstr("tool_use"), "id", jstr("tu_1"), "name", jstr("get_weather"),
			"input", jobj("city", jstr("Paris"))),
	}}
	messages := []AnthropicInMessage{{Role: "assistant", Content: content}}
	opts := AnthropicConvertOptions{NativeToolCalling: true}
	b.ReportAllocs()
	for b.Loop() {
		_ = convertAnthropicToInternal(system, messages, opts)
	}
}
