// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import "strings"

// CodexConfig rewrites a Codex config.toml (passed as text) so it points at the
// fastmlx provider, preserving every unrelated line. It is the only config-file
// updater that works on raw text rather than a parsed map, because the reference
// edits the TOML line by line to keep comments and ordering intact rather than
// round-tripping through a TOML library. Reading the file, taking a backup, and
// writing the result back stay the caller's seams.
//
// The provider section name, model_provider value, display name, and env key
// that the source brands with the upstream name are "fastmlx" /
// "FASTMLX_API_KEY" here; those are the only renames in the Codex path.
func CodexConfig(existing string, c Context) string {
	model := c.Model
	if model == "" {
		model = "select-a-model"
	}

	// Top-level keys to override, kept in order so a fresh file lays them out
	// the same way the reference does.
	type kv struct{ key, val string }
	overrides := []kv{
		{"model", `"` + model + `"`},
		{"model_provider", `"fastmlx"`},
	}
	isReasoning := boolOr(c.Reasoning, isReasoningModel(c.Model))
	if isReasoning {
		overrides = append(overrides, kv{"model_reasoning_effort", `"high"`})
	}

	overrideOf := map[string]string{}
	for _, o := range overrides {
		overrideOf[o.key] = o.val
	}
	// Keys fastmlx manages but is not setting this time: drop them so a config
	// written for a reasoning model is cleaned up when the model is not one.
	managed := map[string]bool{}
	if _, ok := overrideOf["model_reasoning_effort"]; !ok {
		managed["model_reasoning_effort"] = true
	}

	var newLines []string
	inAnySection := false
	inOldSection := false
	seen := map[string]bool{}

	for _, line := range splitLines(existing) {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(stripped, "[") && strings.HasSuffix(stripped, "]") {
			inAnySection = true
			inOldSection = stripped == "[model_providers.fastmlx]"
		}

		if !inAnySection && strings.Contains(stripped, "=") {
			key := strings.TrimSpace(strings.SplitN(stripped, "=", 2)[0])
			if val, ok := overrideOf[key]; ok {
				newLines = append(newLines, key+" = "+val)
				seen[key] = true
				continue
			}
			if managed[key] {
				continue
			}
		}

		if inOldSection {
			continue
		}
		newLines = append(newLines, line)
	}

	// Prepend any override not already present. The reference inserts each at the
	// front in iteration order, so a later key ends up ahead of an earlier one;
	// prepending in that same forward order reproduces the reversed layout.
	for _, o := range overrides {
		if !seen[o.key] {
			newLines = append([]string{o.key + " = " + o.val}, newLines...)
		}
	}

	newLines = append(newLines,
		"\n[model_providers.fastmlx]",
		`name = "fastmlx"`,
		`base_url = "`+c.OpenAIBaseURL()+`"`,
		`env_key = "FASTMLX_API_KEY"`,
	)

	return strings.Join(newLines, "\n") + "\n"
}

// splitLines mirrors Python's str.splitlines for the \n and \r\n cases the
// config files use: it splits on newlines and drops the trailing empty element
// a final newline would otherwise produce.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = strings.TrimSuffix(lines[i], "\r")
	}
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}
