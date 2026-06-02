// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/fastmlx/api"
	"github.com/tamnd/fastmlx/enginecore"
	"github.com/tamnd/fastmlx/pipeline"
	"github.com/tamnd/fastmlx/tokenizer"
)

func newTestApp(t *testing.T, resp string, apiKeys []string) (*App, context.CancelFunc) {
	t.Helper()
	tok := tokenizer.NewMock()
	eng := enginecore.NewBatchedEngine(enginecore.Options{
		ModelName:     "mock-model",
		Tokenizer:     tok,
		Decode:        pipeline.NewMockDecode(tok, resp),
		MaxConcurrent: 8,
	})
	ctx, cancel := context.WithCancel(context.Background())
	eng.Start(ctx)
	app := NewApp(Config{Engine: eng, APIKeys: apiKeys, CORSOrigins: []string{"*"}})
	return app, func() { cancel(); eng.Stop() }
}

func TestChatCompletionNonStreaming(t *testing.T) {
	app, stop := newTestApp(t, "hello from the mock", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":100}`
	res, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(res.Body)
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
	var resp api.ChatCompletionResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "chat.completion" || resp.Model != "mock-model" {
		t.Errorf("envelope mismatch: %+v", resp)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "hello from the mock" {
		t.Errorf("content = %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage.CompletionTokens == 0 || resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Errorf("usage mismatch: %+v", resp.Usage)
	}
}

func TestChatCompletionStreaming(t *testing.T) {
	app, stop := newTestApp(t, "stream me", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":100}`
	res, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if ct := res.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q", ct)
	}

	var content strings.Builder
	var sawRole, sawDone bool
	var finish string
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line := sc.Text()
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			sawDone = true
			break
		}
		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", data, err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		c := chunk.Choices[0]
		if c.Delta.Role == "assistant" {
			sawRole = true
		}
		content.WriteString(c.Delta.Content)
		if c.FinishReason != nil {
			finish = *c.FinishReason
		}
	}
	if !sawRole {
		t.Error("missing role chunk")
	}
	if !sawDone {
		t.Error("missing [DONE]")
	}
	if content.String() != "stream me" {
		t.Errorf("streamed content = %q, want %q", content.String(), "stream me")
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q", finish)
	}
}

func TestAuthRequired(t *testing.T) {
	app, stop := newTestApp(t, "x", []string{"secret-key"})
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`

	// No key -> 401.
	res, _ := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if res.StatusCode != http.StatusUnauthorized {
		t.Errorf("no key: status %d, want 401", res.StatusCode)
	}
	res.Body.Close()

	// Bearer key -> 200.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret-key")
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Errorf("bearer key: status %d, want 200", res.StatusCode)
	}
	res.Body.Close()

	// x-api-key -> 200 (Anthropic SDK style).
	req2, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(body))
	req2.Header.Set("x-api-key", "secret-key")
	req2.Header.Set("Content-Type", "application/json")
	res2, _ := http.DefaultClient.Do(req2)
	if res2.StatusCode != http.StatusOK {
		t.Errorf("x-api-key: status %d, want 200", res2.StatusCode)
	}
	res2.Body.Close()

	// Health bypasses auth.
	hres, _ := http.Get(srv.URL + "/health")
	if hres.StatusCode != http.StatusOK {
		t.Errorf("health: status %d, want 200", hres.StatusCode)
	}
	hres.Body.Close()
}

func TestModelsAndStatus(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var list api.ModelList
	json.NewDecoder(res.Body).Decode(&list)
	res.Body.Close()
	if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "mock-model" {
		t.Errorf("models = %+v", list)
	}

	sres, _ := http.Get(srv.URL + "/api/status")
	var status map[string]any
	json.NewDecoder(sres.Body).Decode(&status)
	sres.Body.Close()
	if status["model"] != "mock-model" {
		t.Errorf("status = %+v", status)
	}
}

func TestCompletionEndpoint(t *testing.T) {
	app, stop := newTestApp(t, "legacy completion", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	body := `{"model":"mock-model","prompt":"once upon a time","max_tokens":50}`
	res, err := http.Post(srv.URL+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var resp api.CompletionResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "text_completion" || resp.Choices[0].Text != "legacy completion" {
		t.Errorf("completion = %+v", resp)
	}
}

func TestCORSPreflight(t *testing.T) {
	app, stop := newTestApp(t, "x", []string{"k"})
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodOptions, srv.URL+"/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://example.com")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	// Preflight bypasses auth and returns 204 with CORS headers.
	if res.StatusCode != http.StatusNoContent {
		t.Errorf("preflight status = %d, want 204", res.StatusCode)
	}
	if got := res.Header.Get("Access-Control-Allow-Origin"); got != "https://example.com" {
		t.Errorf("allow-origin = %q", got)
	}
}

func TestEmbeddingsStub(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader([]byte(`{"model":"m","input":"hi"}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusNotImplemented {
		t.Errorf("embeddings stub status = %d, want 501", res.StatusCode)
	}
}
