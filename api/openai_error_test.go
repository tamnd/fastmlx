// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenAIErrorBodyParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "openai_error_body.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Message    json.RawMessage `json:"message"`
		StatusCode int             `json:"status_code"`
		Param      json.RawMessage `json:"param"`
		Code       json.RawMessage `json:"code"`
		Type       string          `json:"type"`
		Result     json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	jvalOf := func(t *testing.T, raw json.RawMessage) jval {
		t.Helper()
		v, ok := parseOrdered(string(raw))
		if !ok {
			t.Fatalf("not valid JSON: %s", raw)
		}
		return v
	}

	for i, c := range cases {
		if got := StatusToErrorType(c.StatusCode); got != c.Type {
			t.Errorf("case %d: StatusToErrorType(%d) = %q, want %q", i, c.StatusCode, got, c.Type)
		}
		message := jvalOf(t, c.Message)
		param := jvalOf(t, c.Param)
		code := jvalOf(t, c.Code)
		want := jvalOf(t, c.Result)
		got := OpenAIErrorBody(message, c.StatusCode, param, code)
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d:\n got %s\nwant %s", i, g, w)
		}
	}
}

func BenchmarkOpenAIErrorBody(b *testing.B) {
	b.ReportAllocs()
	msg := jstr("Bad field")
	param := jstr("model")
	code := jstr("invalid_value")
	for b.Loop() {
		_ = OpenAIErrorBody(msg, 400, param, code)
	}
}
