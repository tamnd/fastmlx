// SPDX-License-Identifier: MIT OR Apache-2.0

package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// These exercise the /v1/rerank route's request validation against the mock
// backend. The relevance scoring is served by the reranker engine (compute
// backend, later milestone), so a valid request reports not-implemented; the
// validation branches mirror the reference exactly and are checked here.

func postRerank(t *testing.T, srv *httptest.Server, body string) (*http.Response, []byte) {
	t.Helper()
	res, err := http.Post(srv.URL+"/v1/rerank", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(res.Body)
	res.Body.Close()
	return res, b
}

func TestRerankEmptyDocumentsRejected(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, b := postRerank(t, srv, `{"model":"m","query":"q","documents":[]}`)
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
	if !strings.Contains(string(b), "Documents cannot be empty") {
		t.Errorf("body = %s", b)
	}
}

func TestRerankEmptyQueryRejected(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, b := postRerank(t, srv, `{"model":"m","query":"","documents":["a","b"]}`)
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
	if !strings.Contains(string(b), "Query cannot be empty") {
		t.Errorf("body = %s", b)
	}
}

func TestRerankScoringNotImplemented(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	res, b := postRerank(t, srv, `{"model":"m","query":"what is ml?","documents":["ml is ai","weather"]}`)
	if res.StatusCode != http.StatusNotImplemented {
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
	var env struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(b, &env); err != nil {
		t.Fatalf("decode error envelope: %v (%s)", err, b)
	}
	if env.Error.Type != "not_implemented_error" {
		t.Errorf("error type = %q, body = %s", env.Error.Type, b)
	}
}

func TestRerankDictQueryAccepted(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	// A non-empty object query passes the truthiness guard and reaches the
	// compute-gated scoring step.
	res, b := postRerank(t, srv, `{"model":"m","query":{"text":"hi"},"documents":["a"]}`)
	if res.StatusCode != http.StatusNotImplemented {
		t.Fatalf("status %d: %s", res.StatusCode, b)
	}
}
