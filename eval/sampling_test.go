// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type resolveCase struct {
	Base           int    `json:"base"`
	ModelType      string `json:"model_type"`
	EnableThinking bool   `json:"enable_thinking"`
	Out            int    `json:"out"`
}

type promptTextCase struct {
	Messages []fixtureMessage `json:"messages"`
	Out      string           `json:"out"`
}

type thinkCase struct {
	Raws []string `json:"raws"`
	Out  bool     `json:"out"`
}

type samplingKwargsCase struct {
	Sampling       map[string]any `json:"sampling"`
	BaseMaxTokens  int            `json:"base_max_tokens"`
	ModelType      string         `json:"model_type"`
	EnableThinking bool           `json:"enable_thinking"`
	Out            map[string]any `json:"out"`
}

type samplingFixture struct {
	Resolve  []resolveCase        `json:"resolve"`
	Prompt   []promptTextCase     `json:"prompt"`
	Think    []thinkCase          `json:"think"`
	Sampling []samplingKwargsCase `json:"sampling"`
}

func loadSampling(t *testing.T) samplingFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/sampling.json")
	if err != nil {
		t.Fatal(err)
	}
	var f samplingFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// jsonRoundTrip normalizes a value through JSON so its numeric types match a
// decoded fixture (every number becomes float64), letting DeepEqual ignore the
// int-versus-float and 1.0-versus-1 differences between Go and the fixture.
func jsonRoundTrip(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestResolveMaxTokensParity(t *testing.T) {
	for i, c := range loadSampling(t).Resolve {
		if got := ResolveMaxTokens(c.Base, c.ModelType, c.EnableThinking); got != c.Out {
			t.Errorf("ResolveMaxTokens case %d (%d,%q,%v) = %d, want %d",
				i, c.Base, c.ModelType, c.EnableThinking, got, c.Out)
		}
	}
}

func TestPromptTextParity(t *testing.T) {
	for i, c := range loadSampling(t).Prompt {
		messages := make([]Message, len(c.Messages))
		for j, m := range c.Messages {
			messages[j] = Message{Role: m.Role, Content: m.Content}
		}
		if got := PromptText(messages); got != c.Out {
			t.Errorf("PromptText case %d = %q, want %q", i, got, c.Out)
		}
	}
}

func TestHasThinkTagsParity(t *testing.T) {
	for i, c := range loadSampling(t).Think {
		if got := HasThinkTags(c.Raws); got != c.Out {
			t.Errorf("HasThinkTags case %d (%v) = %v, want %v", i, c.Raws, got, c.Out)
		}
	}
}

func TestBuildSamplingKwargsParity(t *testing.T) {
	for i, c := range loadSampling(t).Sampling {
		got := BuildSamplingKwargs(c.Sampling, c.BaseMaxTokens, c.ModelType, c.EnableThinking)
		if !reflect.DeepEqual(jsonRoundTrip(t, got), c.Out) {
			gb, _ := json.Marshal(got)
			wb, _ := json.Marshal(c.Out)
			t.Errorf("BuildSamplingKwargs case %d =\n%s\nwant\n%s", i, gb, wb)
		}
	}
}

func BenchmarkBuildSamplingKwargs(b *testing.B) {
	sampling := map[string]any{"top_p": 0.9, "chat_template_kwargs": map[string]any{"foo": 1}}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildSamplingKwargs(sampling, 512, "", true)
	}
}
