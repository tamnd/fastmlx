// SPDX-License-Identifier: MIT OR Apache-2.0

package engine

// Message is a single chat turn handed to BuildPrompt. Content is the flattened
// text; multimodal parts collapse to text at the API boundary in this stage.
// It is the subset of the chat Message the prompt builder uses.
type Message struct {
	Role             string
	Content          string
	ReasoningContent string
	Name             string
	ToolCallID       string
}

// Tool describes a callable tool advertised to the model. The fields the chat
// template needs are the name, description, and JSON-schema parameters.
type Tool struct {
	Name        string
	Description string
	Parameters  any
}

// PromptOptions tunes prompt construction (chat-template knobs).
type PromptOptions struct {
	AddGenerationPrompt bool
	EnableThinking      *bool
}
