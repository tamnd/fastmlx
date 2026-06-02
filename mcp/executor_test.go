// SPDX-License-Identifier: MIT OR Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"testing"
)

func startedExecutor(t *testing.T) (*Executor, map[string]*fakeTransport) {
	t.Helper()
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"file data"}]}`},
		"db": {tools: `{"tools":[{"name":"query","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"rows"}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	t.Cleanup(m.Stop)
	return NewExecutor(m, 0, 0), fakes
}

func TestExecutorParallelPreservesOrder(t *testing.T) {
	ex, _ := startedExecutor(t)
	calls := []json.RawMessage{
		json.RawMessage(`{"id":"call_1","function":{"name":"fs__read","arguments":"{\"path\":\"/a\"}"}}`),
		json.RawMessage(`{"id":"call_2","function":{"name":"db__query","arguments":"{\"sql\":\"x\"}"}}`),
	}
	results := ex.ExecuteToolCalls(context.Background(), calls, true)
	if len(results) != 2 {
		t.Fatalf("results = %d", len(results))
	}
	if results[0].CallID != "call_1" || string(results[0].Result.Content) != `"file data"` {
		t.Errorf("result[0] = %+v", results[0])
	}
	if results[1].CallID != "call_2" || string(results[1].Result.Content) != `"rows"` {
		t.Errorf("result[1] = %+v", results[1])
	}
}

func TestExecutorSequential(t *testing.T) {
	ex, _ := startedExecutor(t)
	calls := []json.RawMessage{
		json.RawMessage(`{"id":"c","function":{"name":"fs__read","arguments":"{}"}}`),
	}
	results := ex.ExecuteToolCalls(context.Background(), calls, false)
	if len(results) != 1 || results[0].CallID != "c" || results[0].Result.IsError {
		t.Fatalf("results = %+v", results)
	}
}

func TestExecutorArgumentsAsObject(t *testing.T) {
	ex, fakes := startedExecutor(t)
	// arguments given as an object, not a JSON string.
	calls := []json.RawMessage{
		json.RawMessage(`{"id":"c","function":{"name":"fs__read","arguments":{"path":"/b"}}}`),
	}
	ex.ExecuteToolCalls(context.Background(), calls, false)
	if fakes["fs"].lastArgs != `{"path":"/b"}` {
		t.Errorf("forwarded args = %q", fakes["fs"].lastArgs)
	}
}

func TestExecutorBadArgumentsBecomeEmpty(t *testing.T) {
	ex, fakes := startedExecutor(t)
	calls := []json.RawMessage{
		json.RawMessage(`{"id":"c","function":{"name":"fs__read","arguments":"not json"}}`),
	}
	ex.ExecuteToolCalls(context.Background(), calls, false)
	if fakes["fs"].lastArgs != `{}` {
		t.Errorf("forwarded args = %q, want {}", fakes["fs"].lastArgs)
	}
}

func TestExecutorExtractAndValidate(t *testing.T) {
	ex, _ := startedExecutor(t)
	resp := json.RawMessage(`{"choices":[{"message":{"tool_calls":[
		{"id":"1","function":{"name":"fs__read","arguments":"{}"}},
		{"id":"2","function":{"name":"fs__missing","arguments":"{}"}}
	]}}]}`)
	calls, allValid := ex.ExtractAndValidate(resp)
	if len(calls) != 2 {
		t.Fatalf("calls = %d", len(calls))
	}
	if allValid {
		t.Error("expected allValid=false (fs__missing does not exist)")
	}
}

func TestExecutorExtractNoCalls(t *testing.T) {
	ex, _ := startedExecutor(t)
	calls, allValid := ex.ExtractAndValidate(json.RawMessage(`{"choices":[{"message":{}}]}`))
	if len(calls) != 0 || !allValid {
		t.Errorf("calls=%d allValid=%v", len(calls), allValid)
	}
}

func BenchmarkExecutorParallel(b *testing.B) {
	fakes := map[string]*fakeTransport{
		"fs": {tools: `{"tools":[{"name":"read","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"x"}]}`},
		"db": {tools: `{"tools":[{"name":"query","inputSchema":{}}]}`,
			callResult: `{"content":[{"type":"text","text":"y"}]}`},
	}
	m := newManagerWithFakes(twoServerConfig(), fakes)
	m.Start(context.Background())
	defer m.Stop()
	ex := NewExecutor(m, 0, 0)
	calls := []json.RawMessage{
		json.RawMessage(`{"id":"1","function":{"name":"fs__read","arguments":"{}"}}`),
		json.RawMessage(`{"id":"2","function":{"name":"db__query","arguments":"{}"}}`),
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ex.ExecuteToolCalls(context.Background(), calls, true)
	}
}
