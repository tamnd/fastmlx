// SPDX-License-Identifier: MIT OR Apache-2.0

// Package integrations builds the environment and configuration an external
// agent tool (Claude Code, Copilot CLI, and the rest) needs to talk to a local
// fastmlx server. This file holds the GPU-free, side-effect-free core: the
// resolved launch Context and the environment-variable overlays for the
// env-var-style tools. Discovering the tool binary, scrubbing the inherited
// process environment, writing config files, and exec'ing the tool are the
// caller's seams and are not part of this package.
package integrations

import "strconv"

// Context is the resolved set of launch inputs for an integration: where the
// server is, which key and model to use, and the optional per-tier model and
// token overrides. A nil pointer field means the value was not supplied.
type Context struct {
	Host          string
	Port          int
	APIKey        string
	Model         string
	OpusModel     *string
	SonnetModel   *string
	HaikuModel    *string
	ContextWindow *int
	MaxTokens     *int
	ModelType     *string
	Reasoning     *bool
	ToolsProfile  string
	ExtraArgs     []string
}

// BaseURL is the server's root URL.
func (c Context) BaseURL() string {
	return "http://" + c.Host + ":" + strconv.Itoa(c.Port)
}

// OpenAIBaseURL is the OpenAI-compatible API root.
func (c Context) OpenAIBaseURL() string {
	return c.BaseURL() + "/v1"
}

// AuthToken is the bearer token a launched tool should send: the configured API
// key, or a non-empty fallback when the server runs open so the tool still sends
// a token.
func (c Context) AuthToken() string {
	if c.APIKey != "" {
		return c.APIKey
	}
	return "fastmlx"
}

// SupportsImages reports whether the target model accepts images.
func (c Context) SupportsImages() bool {
	return c.ModelType != nil && *c.ModelType == "vlm"
}

// ClaudeEnv is the environment overlay for the Claude Code integration, which
// points the tool at the local server over the Anthropic-compatible API. The
// per-tier model values default to the primary model, and the subagent model
// prefers the smallest available tier; an auto-compact window is set only when a
// context window is supplied.
func ClaudeEnv(c Context) map[string]string {
	env := map[string]string{
		"ANTHROPIC_BASE_URL":                       c.BaseURL(),
		"ANTHROPIC_AUTH_TOKEN":                     c.AuthToken(),
		"ANTHROPIC_API_KEY":                        "",
		"CLAUDE_CODE_ATTRIBUTION_HEADER":           "0",
		"API_TIMEOUT_MS":                           "3000000",
		"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
	}

	opus := pick(c.OpusModel, c.Model)
	sonnet := pick(c.SonnetModel, c.Model)
	haiku := pick(c.HaikuModel, c.Model)

	if opus != "" {
		env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = opus
	}
	if sonnet != "" {
		env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = sonnet
	}
	if haiku != "" {
		env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = haiku
	}

	subagent := firstNonEmpty(haiku, sonnet, opus)
	if subagent != "" {
		env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent
	}
	if intTruthy(c.ContextWindow) {
		env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = strconv.Itoa(*c.ContextWindow)
	}
	return env
}

// CopilotEnv is the environment overlay for the Copilot CLI integration, which
// uses a custom OpenAI-compatible provider over the responses wire API. Model
// and token limits are set only when supplied.
func CopilotEnv(c Context) map[string]string {
	env := map[string]string{
		"COPILOT_PROVIDER_BASE_URL":     c.OpenAIBaseURL(),
		"COPILOT_PROVIDER_TYPE":         "openai",
		"COPILOT_PROVIDER_WIRE_API":     "responses",
		"COPILOT_PROVIDER_BEARER_TOKEN": c.AuthToken(),
	}
	if c.Model != "" {
		env["COPILOT_MODEL"] = c.Model
		env["COPILOT_PROVIDER_MODEL_ID"] = c.Model
		env["COPILOT_PROVIDER_WIRE_MODEL"] = c.Model
	}
	if intTruthy(c.ContextWindow) {
		env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = strconv.Itoa(*c.ContextWindow)
	}
	if intTruthy(c.MaxTokens) {
		env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = strconv.Itoa(*c.MaxTokens)
	}
	return env
}

// pick returns the pointed-to string when it is set and non-empty, otherwise the
// fallback, reproducing Python's `value or fallback`.
func pick(ptr *string, fallback string) string {
	if ptr != nil && *ptr != "" {
		return *ptr
	}
	return fallback
}

// firstNonEmpty returns the first non-empty argument, or "" if all are empty.
func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

// intTruthy reports whether an optional int is set and non-zero, reproducing
// Python's `if value` for an int-or-None field.
func intTruthy(ptr *int) bool {
	return ptr != nil && *ptr != 0
}
