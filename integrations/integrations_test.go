// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import (
	"encoding/json"
	"maps"
	"os"
	"testing"
)

// rawCase mirrors the capture script's IntegrationContext kwargs: optional
// fields are pointers so an absent key stays nil rather than a zero value.
type rawCase struct {
	Host          string   `json:"host"`
	Port          int      `json:"port"`
	APIKey        string   `json:"api_key"`
	Model         string   `json:"model"`
	OpusModel     *string  `json:"opus_model"`
	SonnetModel   *string  `json:"sonnet_model"`
	HaikuModel    *string  `json:"haiku_model"`
	ContextWindow *int     `json:"context_window"`
	MaxTokens     *int     `json:"max_tokens"`
	ModelType     *string  `json:"model_type"`
	Reasoning     *bool    `json:"reasoning"`
	ToolsProfile  string   `json:"tools_profile"`
	ExtraArgs     []string `json:"extra_args"`
}

func (r rawCase) context() Context {
	tools := r.ToolsProfile
	if tools == "" {
		tools = "coding"
	}
	return Context{
		Host:          r.Host,
		Port:          r.Port,
		APIKey:        r.APIKey,
		Model:         r.Model,
		OpusModel:     r.OpusModel,
		SonnetModel:   r.SonnetModel,
		HaikuModel:    r.HaikuModel,
		ContextWindow: r.ContextWindow,
		MaxTokens:     r.MaxTokens,
		ModelType:     r.ModelType,
		Reasoning:     r.Reasoning,
		ToolsProfile:  tools,
		ExtraArgs:     r.ExtraArgs,
	}
}

type integrationsFixture struct {
	Cases   []rawCase `json:"cases"`
	Context []struct {
		BaseURL       string `json:"base_url"`
		OpenAIBaseURL string `json:"openai_base_url"`
		AuthToken     string `json:"auth_token"`
		SupportsImage bool   `json:"supports_images"`
	} `json:"context"`
	Claude  []map[string]string `json:"claude"`
	Copilot []map[string]string `json:"copilot"`
}

func loadFixture(t *testing.T) integrationsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/integrations.json")
	if err != nil {
		t.Fatal(err)
	}
	var f integrationsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestContextPropertiesParity(t *testing.T) {
	f := loadFixture(t)
	for i, rc := range f.Cases {
		c := rc.context()
		want := f.Context[i]
		if c.BaseURL() != want.BaseURL {
			t.Errorf("case %d BaseURL = %q, want %q", i, c.BaseURL(), want.BaseURL)
		}
		if c.OpenAIBaseURL() != want.OpenAIBaseURL {
			t.Errorf("case %d OpenAIBaseURL = %q, want %q", i, c.OpenAIBaseURL(), want.OpenAIBaseURL)
		}
		if c.AuthToken() != want.AuthToken {
			t.Errorf("case %d AuthToken = %q, want %q", i, c.AuthToken(), want.AuthToken)
		}
		if c.SupportsImages() != want.SupportsImage {
			t.Errorf("case %d SupportsImages = %v, want %v", i, c.SupportsImages(), want.SupportsImage)
		}
	}
}

func TestClaudeEnvParity(t *testing.T) {
	f := loadFixture(t)
	for i, rc := range f.Cases {
		got := ClaudeEnv(rc.context())
		if !maps.Equal(got, f.Claude[i]) {
			t.Errorf("case %d ClaudeEnv = %v, want %v", i, got, f.Claude[i])
		}
	}
}

func TestCopilotEnvParity(t *testing.T) {
	f := loadFixture(t)
	for i, rc := range f.Cases {
		got := CopilotEnv(rc.context())
		if !maps.Equal(got, f.Copilot[i]) {
			t.Errorf("case %d CopilotEnv = %v, want %v", i, got, f.Copilot[i])
		}
	}
}

func BenchmarkClaudeEnv(b *testing.B) {
	cw := 32768
	c := Context{Host: "127.0.0.1", Port: 8000, APIKey: "k", Model: "Qwen3-8B", ContextWindow: &cw}
	b.ReportAllocs()
	for b.Loop() {
		_ = ClaudeEnv(c)
	}
}

func BenchmarkCopilotEnv(b *testing.B) {
	cw, mt := 32768, 4096
	c := Context{Host: "127.0.0.1", Port: 8000, Model: "Qwen3-8B", ContextWindow: &cw, MaxTokens: &mt}
	b.ReportAllocs()
	for b.Loop() {
		_ = CopilotEnv(c)
	}
}
