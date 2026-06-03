// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestParseDeepSeekV4ToolCallParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v4_tool_call.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text   string          `json:"text"`
		Error  string          `json:"error"`
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		got, err := ParseDeepSeekV4ToolCall(c.Text)
		if c.Error != "" {
			if err == nil || err.Error() != c.Error {
				t.Errorf("case %d: ParseDeepSeekV4ToolCall(%q) err = %v want %q", i, c.Text, err, c.Error)
			}
			continue
		}
		if err != nil {
			t.Errorf("case %d: ParseDeepSeekV4ToolCall(%q) unexpected err: %v", i, c.Text, err)
			continue
		}
		want, ok := parseOrdered(string(c.Result))
		if !ok {
			t.Fatalf("case %d: fixture result is not valid JSON: %s", i, c.Result)
		}
		if g, w := got.dump(), want.dump(); g != w {
			t.Errorf("case %d: ParseDeepSeekV4ToolCall(%q)\n got %s\nwant %s", i, c.Text, g, w)
		}
	}
}

func BenchmarkParseDeepSeekV4ToolCall(b *testing.B) {
	b.ReportAllocs()
	text := `<｜DSML｜invoke name="get_weather">` + "\n" +
		`<｜DSML｜parameter name="city" string="true">Seoul</｜DSML｜parameter>` + "\n" +
		`<｜DSML｜parameter name="unit" string="false">"celsius"</｜DSML｜parameter>` + "\n" +
		`</｜DSML｜invoke>`
	for b.Loop() {
		_, _ = ParseDeepSeekV4ToolCall(text)
	}
}
