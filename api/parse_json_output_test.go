// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// stubSchemaValidator mirrors the deterministic validator the parity capture
// injected for the json_schema branch: a schema titled "FAIL" is rejected, any
// other schema passes. The real jsonschema library stays a seam, so the test
// exercises ParseJSONOutput's orchestration with a known validator verdict.
type stubSchemaValidator struct{}

func (stubSchemaValidator) Validate(data, schema jval) (bool, string) {
	if f, ok := schema.getField("title"); ok && f.kind == kindString && f.s == "FAIL" {
		return false, "stub: data does not match schema"
	}
	return true, ""
}

func TestParseJSONOutputParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "parse_json_output.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text           string          `json:"text"`
		ResponseFormat json.RawMessage `json:"response_format"`
		Result         struct {
			CleanedText  string          `json:"cleaned_text"`
			Parsed       json.RawMessage `json:"parsed"`
			Valid        bool            `json:"valid"`
			ErrorMessage *string         `json:"error_message"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		rf := decodeResponseFormat(c.ResponseFormat)
		got := ParseJSONOutput(c.Text, rf, stubSchemaValidator{})

		if got.CleanedText != c.Result.CleanedText {
			t.Errorf("case %d: cleaned_text got %q want %q", i, got.CleanedText, c.Result.CleanedText)
		}
		if got.Valid != c.Result.Valid {
			t.Errorf("case %d: valid got %v want %v", i, got.Valid, c.Result.Valid)
		}

		wantErr := ""
		if c.Result.ErrorMessage != nil {
			wantErr = *c.Result.ErrorMessage
		}
		if got.ErrorMessage != wantErr {
			t.Errorf("case %d: error_message got %q want %q", i, got.ErrorMessage, wantErr)
		}

		if isJSONNull(c.Result.Parsed) {
			if got.Parsed != nil {
				t.Errorf("case %d: parsed got %q want nil", i, got.Parsed.dump())
			}
			continue
		}
		want, ok := parseOrdered(string(c.Result.Parsed))
		if !ok {
			t.Fatalf("case %d: fixture parsed is not valid JSON: %s", i, c.Result.Parsed)
		}
		if got.Parsed == nil {
			t.Errorf("case %d: parsed got nil want %q", i, want.dump())
			continue
		}
		if g, w := got.Parsed.dump(), want.dump(); g != w {
			t.Errorf("case %d: parsed got %q want %q", i, g, w)
		}
	}
}

// decodeResponseFormat turns the fixture's response_format value into a
// *ResponseFormat: a JSON null becomes nil, otherwise the type and the raw
// json_schema sub-object are carried over verbatim.
func decodeResponseFormat(raw json.RawMessage) *ResponseFormat {
	if isJSONNull(raw) {
		return nil
	}
	var rf struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &rf); err != nil {
		return nil
	}
	return &ResponseFormat{Type: rf.Type, JSONSchema: rf.JSONSchema}
}

func isJSONNull(raw json.RawMessage) bool {
	return len(raw) == 0 || string(raw) == "null"
}

func BenchmarkParseJSONOutput(b *testing.B) {
	b.ReportAllocs()
	rf := &ResponseFormat{Type: "json_schema", JSONSchema: json.RawMessage(`{"schema":{"title":"PASS","type":"object"}}`)}
	text := `{"a": 1, "b": [2, 3]}`
	v := stubSchemaValidator{}
	for b.Loop() {
		_ = ParseJSONOutput(text, rf, v)
	}
}
