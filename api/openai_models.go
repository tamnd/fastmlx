// SPDX-License-Identifier: MIT OR Apache-2.0

// Package api holds the OpenAI/Anthropic-compatible request and response models
// (exact json tags so existing SDKs are drop-in) and the streaming SSE encoder.
// These types follow the OpenAI API schema.
package api

import (
	"encoding/json"
	"strings"
)

// ChatCompletionRequest is the body of POST /v1/chat/completions, including the
// extensions (top_k, min_p, repetition_penalty, guided_grammar,
// chat_template_kwargs). Sampling fields are pointers so "unset" is
// distinguishable from "zero" for the sampling cascade.
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	MinP                *float64        `json:"min_p,omitempty"`
	RepetitionPenalty   *float64        `json:"repetition_penalty,omitempty"`
	PresencePenalty     *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	N                   *int            `json:"n,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Stop                StopField       `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	GuidedGrammar       string          `json:"guided_grammar,omitempty"`
	ChatTemplateKwargs  map[string]any  `json:"chat_template_kwargs,omitempty"`
	User                string          `json:"user,omitempty"`
}

// StreamOptions mirrors OpenAI stream_options.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage is one input turn. Content is string | content-parts; ContentText
// flattens it to text for this stage.
type ChatMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
}

// ContentText flattens Content to plain text: a JSON string is returned as-is;
// an array of parts concatenates each part's "text" field.
func (m ChatMessage) ContentText() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var out strings.Builder
		for _, p := range parts {
			out.WriteString(p.Text)
		}
		return out.String()
	}
	return ""
}

// Tool is a tool advertised in the request.
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef is the schema of a callable function. Strict is a pointer so the
// Responses tool conversion can carry through a literal false (the reference
// emits "strict" whenever the source value is not null).
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ToolCall is a model-emitted tool invocation.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Index    *int         `json:"index,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the called function name and JSON argument string.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ResponseFormat mirrors OpenAI response_format.
type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// ChatCompletionResponse is the non-streaming chat result.
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"` // "chat.completion"
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

// ChatChoice is one completion alternative.
type ChatChoice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// ResponseMessage is the assistant message in a non-streaming response.
type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionChunk is one streamed SSE event.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"` // "chat.completion.chunk"
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// ChunkChoice is one alternative within a streamed chunk.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta is the incremental content of a streamed chunk.
type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// Usage carries OpenAI token accounting plus prefix-cache fields.
type Usage struct {
	PromptTokens             int                  `json:"prompt_tokens"`
	CompletionTokens         int                  `json:"completion_tokens"`
	TotalTokens              int                  `json:"total_tokens"`
	PromptTokensDetails      *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	CacheCreationInputTokens int                  `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                  `json:"cache_read_input_tokens,omitempty"`
}

// PromptTokensDetails reports cached prompt tokens (OpenAI cache accounting).
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// CompletionRequest mirrors POST /v1/completions (legacy text completions).
type CompletionRequest struct {
	Model       string    `json:"model"`
	Prompt      StopField `json:"prompt"` // string | []string (reuse the flexible decoder)
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	TopK        *int      `json:"top_k,omitempty"`
	Seed        *int64    `json:"seed,omitempty"`
	N           *int      `json:"n,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        StopField `json:"stop,omitempty"`
	User        string    `json:"user,omitempty"`
}

// CompletionResponse is the non-streaming text completion result.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"` // "text_completion"
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// CompletionChoice is one text completion alternative.
type CompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// Model is one served model in /v1/models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the /v1/models envelope.
type ModelList struct {
	Object string  `json:"object"` // "list"
	Data   []Model `json:"data"`
}

// EmbeddingRequest mirrors POST /v1/embeddings.
type EmbeddingRequest struct {
	Model          string    `json:"model"`
	Input          StopField `json:"input"` // string | []string
	EncodingFormat string    `json:"encoding_format,omitempty"`
	Dimensions     *int      `json:"dimensions,omitempty"`
}

// EmbeddingResponse is the embeddings envelope.
type EmbeddingResponse struct {
	Object string          `json:"object"` // "list"
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// EmbeddingData is one embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"` // "embedding"
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}
