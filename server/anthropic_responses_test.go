// SPDX-License-Identifier: MIT OR Apache-2.0

package server

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
)

// These exercise the Anthropic /v1/messages and Responses /v1/responses routes
// end to end against the mock backend, the v0.3 exit integration: the same
// conversion layer the parity tests cover, now driven through the HTTP handlers.

func TestMessagesNonStreaming(t *testing.T) {
	app, stop := newTestApp(t, "hi from the mock", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`
	res, err := http.Post(srv.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(res.Body)
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}

	var resp struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(resp.ID, "msg_") {
		t.Errorf("id = %q, want msg_ prefix", resp.ID)
	}
	if resp.Type != "message" || resp.Role != "assistant" || resp.Model != "mock-model" {
		t.Errorf("envelope mismatch: %+v", resp)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "hi from the mock" {
		t.Errorf("content = %+v", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.OutputTokens == 0 {
		t.Errorf("output_tokens = 0, want > 0")
	}
}

func TestMessagesSystemAndStringContent(t *testing.T) {
	app, stop := newTestApp(t, "ok", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	// system as a string plus a multi-block user message: both flatten cleanly
	// through the conversion layer into the prompt.
	body := `{"model":"mock-model","max_tokens":50,"system":"be terse",` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}]}]}`
	res, err := http.Post(srv.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(res.Body)
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
}

func TestMessagesEmptyMessagesRejected(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, err := http.Post(srv.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"mock-model","max_tokens":10,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want 422", res.StatusCode)
	}
}

func TestMessagesCountTokens(t *testing.T) {
	app, stop := newTestApp(t, "unused", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	count := func(body string) int {
		res, err := http.Post(srv.URL+"/v1/messages/count_tokens", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(res.Body)
			t.Fatalf("status %d: %s", res.StatusCode, b)
		}
		var out struct {
			InputTokens int `json:"input_tokens"`
		}
		if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out.InputTokens
	}

	short := count(`{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`)
	if short <= 0 {
		t.Fatalf("input_tokens = %d, want > 0", short)
	}
	// A longer conversation must count more prompt tokens than a shorter one.
	long := count(`{"model":"mock-model","messages":[{"role":"user","content":"hello there, this is a much longer prompt"}]}`)
	if long <= short {
		t.Errorf("long count %d not greater than short count %d", long, short)
	}
}

func TestMessagesCountTokensEmptyRejected(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, err := http.Post(srv.URL+"/v1/messages/count_tokens", "application/json",
		strings.NewReader(`{"model":"mock-model","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want 422", res.StatusCode)
	}
}

func TestMessagesStreaming(t *testing.T) {
	app, stop := newTestApp(t, "stream me", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`
	res, err := http.Post(srv.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if ct := res.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q", ct)
	}

	var events []string
	var text strings.Builder
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line := sc.Text()
		if ev, ok := strings.CutPrefix(line, "event: "); ok {
			events = append(events, ev)
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var payload struct {
			Type  string `json:"type"`
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			t.Fatalf("bad event payload %q: %v", data, err)
		}
		if payload.Type == "content_block_delta" && payload.Delta.Type == "text_delta" {
			text.WriteString(payload.Delta.Text)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}

	want := []string{"message_start", "content_block_start"}
	for i, ev := range want {
		if i >= len(events) || events[i] != ev {
			t.Fatalf("event[%d] = %q, want %q (events: %v)", i, get(events, i), ev, events)
		}
	}
	last := events[len(events)-1]
	if last != "message_stop" {
		t.Errorf("last event = %q, want message_stop", last)
	}
	if !slices.Contains(events, "message_delta") || !slices.Contains(events, "content_block_stop") {
		t.Errorf("missing terminal events: %v", events)
	}
	if text.String() != "stream me" {
		t.Errorf("streamed text = %q, want %q", text.String(), "stream me")
	}
}

func TestResponsesNonStreaming(t *testing.T) {
	app, stop := newTestApp(t, "responses output", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","input":"say hi","instructions":"be brief","temperature":0.7}`
	res, err := http.Post(srv.URL+"/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(res.Body)
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}

	var resp struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Status string `json:"status"`
		Model  string `json:"model"`
		Output []struct {
			Type    string `json:"type"`
			ID      string `json:"id"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
		Temperature float64 `json:"temperature"`
	}
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(resp.ID, "resp_") {
		t.Errorf("id = %q, want resp_ prefix", resp.ID)
	}
	if resp.Object != "response" || resp.Status != "completed" || resp.Model != "mock-model" {
		t.Errorf("envelope mismatch: %+v", resp)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != "message" {
		t.Fatalf("output = %+v", resp.Output)
	}
	if !strings.HasPrefix(resp.Output[0].ID, "msg_") {
		t.Errorf("output id = %q, want msg_ prefix", resp.Output[0].ID)
	}
	if len(resp.Output[0].Content) != 1 || resp.Output[0].Content[0].Text != "responses output" {
		t.Errorf("output content = %+v", resp.Output[0].Content)
	}
	if resp.Temperature != 0.7 {
		t.Errorf("temperature echo = %v, want 0.7", resp.Temperature)
	}
	if resp.Usage.TotalTokens != resp.Usage.InputTokens+resp.Usage.OutputTokens {
		t.Errorf("usage mismatch: %+v", resp.Usage)
	}
}

func TestResponsesStreamingNotImplemented(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","input":"hi","stream":true}`
	res, err := http.Post(srv.URL+"/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", res.StatusCode)
	}
}

func TestResponsesEmptyInputRejected(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, err := http.Post(srv.URL+"/v1/responses", "application/json",
		strings.NewReader(`{"model":"mock-model"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want 422", res.StatusCode)
	}
}

func get(s []string, i int) string {
	if i < len(s) {
		return s[i]
	}
	return "<none>"
}
