// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type toolTemplateCase struct {
	In  json.RawMessage `json:"in"`
	Out json.RawMessage `json:"out"`
}

type toolTemplateFixture struct {
	Convert []toolTemplateCase `json:"convert"`
	Enrich  []toolTemplateCase `json:"enrich"`
	Restore []toolTemplateCase `json:"restore"`
}

func loadToolTemplateFixture(t *testing.T) toolTemplateFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/tool_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var f toolTemplateFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// parseFixture decodes a raw fixture value into the order-preserving jval model.
func parseFixture(t *testing.T, raw json.RawMessage) jval {
	t.Helper()
	v, ok := parseOrdered(string(raw))
	if !ok {
		t.Fatalf("could not parse fixture value %s", raw)
	}
	return v
}

func TestConvertToolsForTemplateParity(t *testing.T) {
	for i, c := range loadToolTemplateFixture(t).Convert {
		in := parseFixture(t, c.In)
		want := parseFixture(t, c.Out).dump()
		if got := ConvertToolsForTemplate(in).dump(); got != want {
			t.Errorf("convert[%d]: got %s, want %s", i, got, want)
		}
	}
}

func TestEnrichToolParamsForGemma4Parity(t *testing.T) {
	for i, c := range loadToolTemplateFixture(t).Enrich {
		in := parseFixture(t, c.In)
		want := parseFixture(t, c.Out).dump()
		if got := EnrichToolParamsForGemma4(in).dump(); got != want {
			t.Errorf("enrich[%d]: got %s, want %s", i, got, want)
		}
	}
}

func TestRestoreGemma4ParamNamesParity(t *testing.T) {
	for i, c := range loadToolTemplateFixture(t).Restore {
		in := parseFixture(t, c.In)
		want := parseFixture(t, c.Out).dump()
		if got := RestoreGemma4ParamNames(in).dump(); got != want {
			t.Errorf("restore[%d]: got %s, want %s", i, got, want)
		}
	}
}

func TestFormatToolCallForMessage(t *testing.T) {
	tc := ToolCall{ID: "call_1", Type: "function",
		Function: FunctionCall{Name: "get_weather", Arguments: `{"city": "Paris"}`}}
	want := `{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}`
	if got := FormatToolCallForMessage(tc).dump(); got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func BenchmarkEnrichToolParamsForGemma4(b *testing.B) {
	raw := `[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"description":{"type":"string"},"city":{"type":"string"}},"required":["description","city"]}}}]`
	tools, _ := parseOrdered(raw)
	b.ReportAllocs()
	for b.Loop() {
		_ = EnrichToolParamsForGemma4(tools).dump()
	}
}
