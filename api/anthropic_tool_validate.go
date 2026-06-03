// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"errors"
	"strings"
)

// errToolSchemaOrType is raised when an Anthropic tool carries neither an
// input_schema nor a type, mirroring the reference validator's message verbatim.
var errToolSchemaOrType = errors.New(
	"AnthropicTool requires either 'input_schema' (user-defined tool) or 'type' (Anthropic server-side tool).")

// toolHasInputSchema reports whether the tool carries a usable input_schema. A
// missing field or an explicit JSON null both count as absent, matching the
// reference's Optional[dict] default of None.
func toolHasInputSchema(t AnthropicTool) bool {
	trimmed := strings.TrimSpace(string(t.InputSchema))
	return trimmed != "" && trimmed != "null"
}

// ValidateAnthropicTool ports AnthropicTool._require_schema_or_type: a tool must
// be either a user-defined tool (carrying input_schema) or an Anthropic
// server-side tool (carrying a versioned type such as web_search_20250305).
// A tool with neither is rejected. It returns errToolSchemaOrType on failure and
// nil otherwise.
func ValidateAnthropicTool(t AnthropicTool) error {
	if !toolHasInputSchema(t) && t.Type == nil {
		return errToolSchemaOrType
	}
	return nil
}

// ValidateAnthropicTools applies ValidateAnthropicTool to each tool, returning
// the first failure, matching Pydantic's per-item after-validation.
func ValidateAnthropicTools(tools []AnthropicTool) error {
	for _, t := range tools {
		if err := ValidateAnthropicTool(t); err != nil {
			return err
		}
	}
	return nil
}
