// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"sync"
)

// defaultMaxParallel bounds concurrent tool executions when running in parallel.
const defaultMaxParallel = 5

// Executor runs tool calls from a model response against the manager. It can run
// calls in parallel (with a concurrency limit) or sequentially, and it can
// validate that the requested tools exist before running them.
type Executor struct {
	manager        *Manager
	maxParallel    int
	defaultTimeout float64
}

// NewExecutor builds an executor over a manager. A non-positive maxParallel uses
// the default; a non-positive timeout uses the manager's default timeout.
func NewExecutor(manager *Manager, maxParallel int, defaultTimeout float64) *Executor {
	if maxParallel <= 0 {
		maxParallel = defaultMaxParallel
	}
	if defaultTimeout <= 0 {
		defaultTimeout = manager.config.DefaultTimeout
	}
	return &Executor{manager: manager, maxParallel: maxParallel, defaultTimeout: defaultTimeout}
}

// ExecutedCall pairs a tool result with the originating tool-call id.
type ExecutedCall struct {
	Result ToolResult
	CallID string
}

// ExecuteToolCalls runs a batch of OpenAI tool-call objects. When parallel is
// set the calls run concurrently up to maxParallel; otherwise they run in order.
// Results preserve input order in both modes.
func (e *Executor) ExecuteToolCalls(ctx context.Context, toolCalls []json.RawMessage, parallel bool) []ExecutedCall {
	if len(toolCalls) == 0 {
		return nil
	}
	if parallel {
		return e.executeParallel(ctx, toolCalls)
	}
	return e.executeSequential(ctx, toolCalls)
}

func (e *Executor) executeSequential(ctx context.Context, toolCalls []json.RawMessage) []ExecutedCall {
	out := make([]ExecutedCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		out = append(out, e.runOne(ctx, tc))
	}
	return out
}

func (e *Executor) executeParallel(ctx context.Context, toolCalls []json.RawMessage) []ExecutedCall {
	out := make([]ExecutedCall, len(toolCalls))
	sem := make(chan struct{}, e.maxParallel)
	var wg sync.WaitGroup
	for i, tc := range toolCalls {
		wg.Add(1)
		go func(i int, tc json.RawMessage) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			out[i] = e.runOne(ctx, tc)
		}(i, tc)
	}
	wg.Wait()
	return out
}

// runOne routes one tool call to its server and returns the result with the call
// id.
func (e *Executor) runOne(ctx context.Context, toolCall json.RawMessage) ExecutedCall {
	id, fullName, args := parseToolCall(toolCall)
	res := e.manager.ExecuteTool(ctx, fullName, args, e.defaultTimeout)
	return ExecutedCall{Result: res, CallID: id}
}

// ExtractAndValidate pulls the tool calls out of a model response and reports
// whether every requested tool exists in a connected server.
func (e *Executor) ExtractAndValidate(response json.RawMessage) ([]json.RawMessage, bool) {
	calls := extractToolCalls(response)
	if len(calls) == 0 {
		return nil, true
	}
	allValid := true
	for _, tc := range calls {
		_, name, _ := parseToolCall(tc)
		if !e.manager.HasTool(name) {
			allValid = false
		}
	}
	return calls, allValid
}

// parseToolCall reads the id, function name, and arguments from an OpenAI
// tool-call object. Arguments are decoded from the JSON string the model emits
// (an object passes through, an unparseable string becomes an empty object).
func parseToolCall(toolCall json.RawMessage) (id, name string, arguments json.RawMessage) {
	var tc struct {
		ID       string `json:"id"`
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(toolCall, &tc); err != nil {
		return "", "", json.RawMessage("{}")
	}
	return tc.ID, tc.Function.Name, normalizeArguments(tc.Function.Arguments)
}

// normalizeArguments turns a tool call's arguments into a JSON object. The model
// usually emits a JSON-encoded string; an object passes through, and anything
// unparseable becomes an empty object.
func normalizeArguments(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage("{}")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return json.RawMessage("{}")
		}
		if json.Valid([]byte(s)) {
			return json.RawMessage(s)
		}
		return json.RawMessage("{}")
	}
	if json.Valid(raw) {
		return raw
	}
	return json.RawMessage("{}")
}

// extractToolCalls returns the tool calls in choices[0].message.tool_calls, or
// nil.
func extractToolCalls(response json.RawMessage) []json.RawMessage {
	var r struct {
		Choices []struct {
			Message struct {
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(response, &r); err != nil || len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].Message.ToolCalls
}
