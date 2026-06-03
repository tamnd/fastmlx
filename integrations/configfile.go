// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import "regexp"

// This file holds the config-file integrations' pure cores: the updater
// functions that merge a fastmlx provider into an existing tool config. Each
// takes the parsed existing config and returns it mutated, leaving unrelated
// keys intact, exactly as the reference's in-place updater does. Reading the
// file, taking a timestamped backup, and writing it back are the caller's seams.
// The provider is keyed "fastmlx" where the source brands it with its own name;
// that rename, and the matching default-model prefix, are the only changes.

// reasoningModelRe matches the reasoning-model name cues; the reference
// lower-cases the model id first, so a case-insensitive match is equivalent.
var reasoningModelRe = regexp.MustCompile(`(?i)\b(thinking|o1|o3|r1)\b`)

// isReasoningModel reports whether a model id names a reasoning model.
func isReasoningModel(model string) bool {
	return reasoningModelRe.MatchString(model)
}

// OpenClawConfig merges the fastmlx provider into an OpenClaw config under
// models.providers, sets it as the default agent model when a model is given,
// and records the tools profile.
func OpenClawConfig(config map[string]any, c Context) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	providers := childMap(childMap(config, "models"), "providers")

	provider := map[string]any{
		"baseUrl": c.OpenAIBaseURL(),
		"apiKey":  c.AuthToken(),
		"api":     "openai-completions",
	}
	if c.Model != "" {
		entry := map[string]any{
			"id":        c.Model,
			"name":      c.Model,
			"api":       "openai-completions",
			"reasoning": boolOr(c.Reasoning, false),
			"input":     inputModalities(c.SupportsImages()),
			"cost":      zeroCost(),
		}
		if intTruthy(c.ContextWindow) {
			entry["contextWindow"] = *c.ContextWindow
		}
		if intTruthy(c.MaxTokens) {
			entry["maxTokens"] = *c.MaxTokens
		}
		provider["models"] = []any{entry}
	}
	providers["fastmlx"] = provider

	if c.Model != "" {
		model := childMap(childMap(childMap(config, "agents"), "defaults"), "model")
		model["primary"] = "fastmlx/" + c.Model
	}
	childMap(config, "tools")["profile"] = c.ToolsProfile
	return config
}

// OpenClawExecApprovals sets the exec-approvals defaults from the tools profile:
// the coding and full profiles get unrestricted exec, others an allowlist.
func OpenClawExecApprovals(config map[string]any, toolsProfile string) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	defaults := childMap(config, "defaults")
	if toolsProfile == "coding" || toolsProfile == "full" {
		defaults["security"] = "full"
		defaults["ask"] = "off"
	} else {
		defaults["security"] = "allowlist"
		defaults["ask"] = "on-miss"
	}
	return config
}

// OpenCodeConfig merges the fastmlx provider into an OpenCode config under
// provider, with model modality metadata, and sets it as the default model.
func OpenCodeConfig(config map[string]any, c Context) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	provider := map[string]any{
		"npm":  "@ai-sdk/openai-compatible",
		"name": "fastmlx",
		"options": map[string]any{
			"baseURL": c.OpenAIBaseURL(),
		},
	}
	if c.APIKey != "" {
		provider["options"].(map[string]any)["apiKey"] = c.APIKey
	}
	if c.Model != "" {
		entry := map[string]any{
			"name":       c.Model,
			"modalities": modalitiesFor(c.ModelType),
		}
		if c.SupportsImages() {
			entry["attachment"] = true
		}
		if intTruthy(c.ContextWindow) {
			entry["limit"] = map[string]any{
				"context": *c.ContextWindow,
				"output":  intOr(c.MaxTokens, *c.ContextWindow),
			}
		}
		provider["models"] = map[string]any{c.Model: entry}
	}
	childMap(config, "provider")["fastmlx"] = provider
	if c.Model != "" {
		config["model"] = "fastmlx/" + c.Model
	}
	return config
}

// PiModels merges the fastmlx provider into Pi's models.json under providers.
func PiModels(config map[string]any, c Context) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	provider := map[string]any{
		"baseUrl":    c.OpenAIBaseURL(),
		"api":        "openai-completions",
		"apiKey":     c.AuthToken(),
		"authHeader": true,
	}
	if c.Model != "" {
		reasoning := boolOr(c.Reasoning, isReasoningModel(c.Model))
		entry := map[string]any{
			"id":        c.Model,
			"name":      c.Model,
			"reasoning": reasoning,
			"input":     inputModalities(c.SupportsImages()),
			"cost":      zeroCost(),
		}
		if intTruthy(c.ContextWindow) {
			entry["contextWindow"] = *c.ContextWindow
		}
		if intTruthy(c.MaxTokens) {
			entry["maxTokens"] = *c.MaxTokens
		}
		provider["models"] = []any{entry}
	}
	childMap(config, "providers")["fastmlx"] = provider
	return config
}

// PiSettings sets fastmlx as Pi's default provider and, when given, the default
// model in settings.json.
func PiSettings(config map[string]any, c Context) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	config["defaultProvider"] = "fastmlx"
	if c.Model != "" {
		config["defaultModel"] = c.Model
	}
	return config
}

// hermesMinContextLength is the smallest context Hermes Agent will start with;
// a smaller reported window is bumped up to this floor.
const hermesMinContextLength = 64000

// HermesConfig merges the fastmlx provider into a Hermes config (config.yaml,
// parsed to a map) and points the model block at it. Context window and max
// tokens are gated on presence, not truthiness, so a reported zero still writes
// the bumped-up minimum; absent ones are cleared from a stale model block.
func HermesConfig(config map[string]any, c Context) map[string]any {
	if config == nil {
		config = map[string]any{}
	}
	providers := childMap(config, "providers")

	provider, ok := providers["fastmlx"].(map[string]any)
	if !ok {
		provider = map[string]any{}
	}
	provider["name"] = "fastmlx"
	provider["base_url"] = c.OpenAIBaseURL()
	provider["api_key"] = c.AuthToken()
	provider["api_mode"] = "chat_completions"
	if c.Model != "" {
		provider["default_model"] = c.Model
	}
	providers["fastmlx"] = provider

	model, ok := config["model"].(map[string]any)
	if !ok {
		model = map[string]any{}
	}
	for _, stale := range []string{"base_url", "api_key", "api", "api_mode", "transport"} {
		delete(model, stale)
	}
	model["provider"] = "fastmlx"
	if c.Model != "" {
		model["default"] = c.Model
	}
	if c.ContextWindow != nil {
		model["context_length"] = max(*c.ContextWindow, hermesMinContextLength)
	} else {
		delete(model, "context_length")
	}
	if c.MaxTokens != nil {
		model["max_tokens"] = *c.MaxTokens
	} else {
		delete(model, "max_tokens")
	}
	config["model"] = model
	return config
}

// childMap returns the map stored at key, creating an empty one when the key is
// absent or not a map, reproducing dict.setdefault for the nested-config case.
func childMap(parent map[string]any, key string) map[string]any {
	if existing, ok := parent[key].(map[string]any); ok {
		return existing
	}
	m := map[string]any{}
	parent[key] = m
	return m
}

func inputModalities(image bool) []any {
	if image {
		return []any{"text", "image"}
	}
	return []any{"text"}
}

func modalitiesFor(modelType *string) map[string]any {
	input := []any{"text"}
	if modelType != nil && *modelType == "vlm" {
		input = append(input, "image")
	}
	return map[string]any{"input": input, "output": []any{"text"}}
}

func zeroCost() map[string]any {
	return map[string]any{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
}

// boolOr returns the pointed-to bool when set, else the fallback, reproducing
// `bool(value) if value is not None else fallback`.
func boolOr(ptr *bool, fallback bool) bool {
	if ptr != nil {
		return *ptr
	}
	return fallback
}

// intOr returns the pointed-to int when truthy, else the fallback, reproducing
// Python's `value or fallback` for an int-or-None.
func intOr(ptr *int, fallback int) int {
	if ptr != nil && *ptr != 0 {
		return *ptr
	}
	return fallback
}
