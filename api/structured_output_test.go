// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// The fixtures in testdata/parity/jsonout.json are captured from the Python
// reference helpers. ExtractJSONFromText parity is checked by re-serializing
// the extracted value Python-style and comparing it to the reference's
// json.dumps of the same result. BuildJSONSystemPrompt parity is byte-exact,
// including the indent=2 schema block.

type extractFixture struct {
	Input  string  `json:"input"`
	Result *string `json:"result"`
}

type promptFixture struct {
	Format json.RawMessage `json:"format"`
	Result *string         `json:"result"`
}

type jsonOutFixtures struct {
	Extract []extractFixture `json:"extract"`
	Prompt  []promptFixture  `json:"prompt"`
}

func loadJSONOutFixtures(t testing.TB) jsonOutFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/jsonout.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx jsonOutFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Extract) == 0 || len(fx.Prompt) == 0 {
		t.Fatal("fixtures missing a group")
	}
	return fx
}

func TestExtractJSONFromTextParity(t *testing.T) {
	for _, fx := range loadJSONOutFixtures(t).Extract {
		t.Run(fx.Input, func(t *testing.T) {
			v, ok := ExtractJSONFromText(fx.Input)
			if fx.Result == nil {
				if ok {
					t.Errorf("ExtractJSONFromText(%q) = %q, want no result", fx.Input, v.dump())
				}
				return
			}
			if !ok {
				t.Fatalf("ExtractJSONFromText(%q) found nothing, want %q", fx.Input, *fx.Result)
			}
			if got := v.dump(); got != *fx.Result {
				t.Errorf("ExtractJSONFromText(%q)\n got  %q\n want %q", fx.Input, got, *fx.Result)
			}
		})
	}
}

func TestBuildJSONSystemPromptParity(t *testing.T) {
	for _, fx := range loadJSONOutFixtures(t).Prompt {
		var rf *ResponseFormat
		if len(fx.Format) > 0 && string(fx.Format) != "null" {
			rf = &ResponseFormat{}
			// The fixture format object carries type plus a nested json_schema;
			// decode into the wire type the same way a request would.
			var raw struct {
				Type       string          `json:"type"`
				JSONSchema json.RawMessage `json:"json_schema"`
			}
			if err := json.Unmarshal(fx.Format, &raw); err != nil {
				t.Fatalf("decode format %s: %v", fx.Format, err)
			}
			rf.Type = raw.Type
			rf.JSONSchema = raw.JSONSchema
		}
		want := ""
		if fx.Result != nil {
			want = *fx.Result
		}
		t.Run(string(fx.Format), func(t *testing.T) {
			if got := BuildJSONSystemPrompt(rf); got != want {
				t.Errorf("BuildJSONSystemPrompt(%s)\n got  %q\n want %q", fx.Format, got, want)
			}
		})
	}
}

func BenchmarkExtractJSONFromText(b *testing.B) {
	const in = "here is the result you asked for:\n```json\n" +
		`{"city": "Paris", "temp": 21, "conditions": ["sunny", "mild"]}` +
		"\n```\nlet me know if you need more."
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ExtractJSONFromText(in)
	}
}
