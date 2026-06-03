// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type capabilitiesFixture struct {
	Vision []struct {
		Config string `json:"config"`
		Result bool   `json:"result"`
	} `json:"vision"`
	Arch []struct {
		Architectures []string `json:"architectures"`
		Result        bool     `json:"result"`
	} `json:"arch"`
	Context []struct {
		Config          *string `json:"config"`
		TokenizerConfig *string `json:"tokenizer_config"`
		Result          *int    `json:"result"`
	} `json:"context"`
	Thinking []struct {
		Template string `json:"template"`
		Result   *bool  `json:"result"`
	} `json:"thinking"`
	Preserve []struct {
		Template string `json:"template"`
		Result   *bool  `json:"result"`
	} `json:"preserve"`
}

func loadCapabilitiesFixture(t *testing.T) capabilitiesFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/capabilities.json")
	if err != nil {
		t.Fatal(err)
	}
	var f capabilitiesFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// mustNumberMap decodes a fixture's config string the same way the production
// code reads config.json, so float-vs-int rejection is exercised faithfully.
func mustNumberMap(t *testing.T, s *string) map[string]any {
	t.Helper()
	if s == nil {
		return nil
	}
	m, err := decodeNumberMap([]byte(*s))
	if err != nil {
		t.Fatalf("decodeNumberMap(%s): %v", *s, err)
	}
	return m
}

func boolPtrEq(a, b *bool) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func TestHasVisionSubconfigParity(t *testing.T) {
	for _, c := range loadCapabilitiesFixture(t).Vision {
		config := mustNumberMap(t, &c.Config)
		if got := HasVisionSubconfig(config); got != c.Result {
			t.Errorf("HasVisionSubconfig(%s) = %v, want %v", c.Config, got, c.Result)
		}
	}
}

func TestArchitectureIndicatesCausalLMParity(t *testing.T) {
	for _, c := range loadCapabilitiesFixture(t).Arch {
		if got := ArchitectureIndicatesCausalLM(c.Architectures); got != c.Result {
			t.Errorf("ArchitectureIndicatesCausalLM(%v) = %v, want %v", c.Architectures, got, c.Result)
		}
	}
}

func TestContextLengthFromConfigsParity(t *testing.T) {
	want0 := func(p *int) int {
		if p == nil {
			return 0
		}
		return *p
	}
	for _, c := range loadCapabilitiesFixture(t).Context {
		config := mustNumberMap(t, c.Config)
		tc := mustNumberMap(t, c.TokenizerConfig)
		if got := ContextLengthFromConfigs(config, tc); got != want0(c.Result) {
			t.Errorf("ContextLengthFromConfigs(%v, %v) = %d, want %d",
				c.Config, c.TokenizerConfig, got, want0(c.Result))
		}
	}
}

func TestDetectThinkingDefaultParity(t *testing.T) {
	for _, c := range loadCapabilitiesFixture(t).Thinking {
		if got := DetectThinkingDefault(c.Template); !boolPtrEq(got, c.Result) {
			t.Errorf("DetectThinkingDefault(%q) = %v, want %v", c.Template, strBoolPtr(got), strBoolPtr(c.Result))
		}
	}
}

func TestDetectPreserveThinkingParity(t *testing.T) {
	for _, c := range loadCapabilitiesFixture(t).Preserve {
		if got := DetectPreserveThinking(c.Template); !boolPtrEq(got, c.Result) {
			t.Errorf("DetectPreserveThinking(%q) = %v, want %v", c.Template, strBoolPtr(got), strBoolPtr(c.Result))
		}
	}
}

func strBoolPtr(b *bool) string {
	if b == nil {
		return "nil"
	}
	if *b {
		return "true"
	}
	return "false"
}

// TestModelTemplateText exercises the disk seam: jinja first, then the
// tokenizer_config chat_template, then "".
func TestModelTemplateText(t *testing.T) {
	dir := t.TempDir()
	if got := ModelTemplateText(dir); got != "" {
		t.Errorf("empty dir = %q, want \"\"", got)
	}

	tcDir := filepath.Join(dir, "tc")
	if err := os.MkdirAll(tcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tcDir, "tokenizer_config.json"),
		[]byte(`{"chat_template":"from tc"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := ModelTemplateText(tcDir); got != "from tc" {
		t.Errorf("tc fallback = %q, want \"from tc\"", got)
	}
	if err := os.WriteFile(filepath.Join(tcDir, "chat_template.jinja"),
		[]byte("from jinja"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := ModelTemplateText(tcDir); got != "from jinja" {
		t.Errorf("jinja precedence = %q, want \"from jinja\"", got)
	}
}

type detectorsFixture struct {
	UnsupportedArchCount int `json:"unsupported_arch_count"`
	UnsupportedTypeCount int `json:"unsupported_type_count"`
	STPipeline           []struct {
		Modules string `json:"modules"`
		Result  bool   `json:"result"`
	} `json:"st_pipeline"`
	Unsupported []struct {
		Config string `json:"config"`
		Result bool   `json:"result"`
	} `json:"unsupported"`
	Reranker []struct {
		Name   string `json:"name"`
		Result bool   `json:"result"`
	} `json:"reranker"`
	Embedding []struct {
		Name   string `json:"name"`
		Result bool   `json:"result"`
	} `json:"embedding"`
}

func loadDetectorsFixture(t *testing.T) detectorsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/detectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var f detectorsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestUnsupportedSetsEmpty(t *testing.T) {
	f := loadDetectorsFixture(t)
	if got := len(UnsupportedArchitectures); got != f.UnsupportedArchCount {
		t.Errorf("UnsupportedArchitectures len = %d, want %d", got, f.UnsupportedArchCount)
	}
	if got := len(UnsupportedModelTypes); got != f.UnsupportedTypeCount {
		t.Errorf("UnsupportedModelTypes len = %d, want %d", got, f.UnsupportedTypeCount)
	}
}

func TestHasSentenceTransformersEmbeddingPipelineParity(t *testing.T) {
	for _, c := range loadDetectorsFixture(t).STPipeline {
		var modules any
		if err := json.Unmarshal([]byte(c.Modules), &modules); err != nil {
			t.Fatalf("unmarshal %s: %v", c.Modules, err)
		}
		if got := HasSentenceTransformersEmbeddingPipeline(modules); got != c.Result {
			t.Errorf("HasSentenceTransformersEmbeddingPipeline(%s) = %v, want %v", c.Modules, got, c.Result)
		}
	}
}

func TestIsUnsupportedModelParity(t *testing.T) {
	for _, c := range loadDetectorsFixture(t).Unsupported {
		var config map[string]any
		if err := json.Unmarshal([]byte(c.Config), &config); err != nil {
			t.Fatalf("unmarshal %s: %v", c.Config, err)
		}
		if got := IsUnsupportedModel(config); got != c.Result {
			t.Errorf("IsUnsupportedModel(%s) = %v, want %v", c.Config, got, c.Result)
		}
	}
}

func TestIsCausalLMRerankerParity(t *testing.T) {
	for _, c := range loadDetectorsFixture(t).Reranker {
		if got := IsCausalLMReranker(c.Name); got != c.Result {
			t.Errorf("IsCausalLMReranker(%q) = %v, want %v", c.Name, got, c.Result)
		}
	}
}

func TestIsCausalLMEmbeddingParity(t *testing.T) {
	for _, c := range loadDetectorsFixture(t).Embedding {
		if got := IsCausalLMEmbedding(c.Name); got != c.Result {
			t.Errorf("IsCausalLMEmbedding(%q) = %v, want %v", c.Name, got, c.Result)
		}
	}
}

func BenchmarkContextLengthFromConfigs(b *testing.B) {
	config, _ := decodeNumberMap([]byte(`{"text_config":{"max_position_embeddings":131072}}`))
	b.ReportAllocs()
	for b.Loop() {
		_ = ContextLengthFromConfigs(config, nil)
	}
}

func BenchmarkDetectThinkingDefault(b *testing.B) {
	const tmpl = "{% if enable_thinking is false %}suppress{% endif %}"
	b.ReportAllocs()
	for b.Loop() {
		_ = DetectThinkingDefault(tmpl)
	}
}
