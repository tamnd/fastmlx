// SPDX-License-Identifier: MIT OR Apache-2.0

package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// These exercise the /v1/embeddings route's request validation and
// normalization against the mock backend. The embedding computation is served by
// the embedding engine (compute backend, later milestone), so a valid request
// reports not-implemented; the validation branches mirror the reference.

func postEmbeddings(t *testing.T, srv *httptest.Server, body string) *http.Response {
	t.Helper()
	res, err := http.Post(srv.URL+"/v1/embeddings", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestEmbeddingsValidation(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	cases := []struct {
		name string
		body string
		code int
		msg  string
	}{
		{"missing model", `{"input":"hi"}`, 422, "model is required"},
		{"bad encoding_format", `{"model":"m","input":"hi","encoding_format":"hex"}`, 422, "encoding_format"},
		{"neither input nor items", `{"model":"m"}`, 422, "Either input or items must be provided"},
		{"both input and items", `{"model":"m","input":"hi","items":[{"text":"a"}]}`, 422, "cannot be provided together"},
		{"empty items list", `{"model":"m","items":[]}`, 422, "items cannot be empty"},
		{"item without text or image", `{"model":"m","items":[{}]}`, 422, "must include text or image"},
		{"item with unknown field", `{"model":"m","items":[{"text":"a","bogus":1}]}`, 422, "invalid embedding item"},
		{"input wrong type", `{"model":"m","input":123}`, 422, "string or a list of strings"},
		{"empty input list", `{"model":"m","input":[]}`, 400, "Input cannot be empty"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := postEmbeddings(t, srv, c.body)
			defer res.Body.Close()
			buf := make([]byte, 4096)
			n, _ := res.Body.Read(buf)
			body := string(buf[:n])
			if res.StatusCode != c.code {
				t.Fatalf("status %d, want %d: %s", res.StatusCode, c.code, body)
			}
			if !strings.Contains(body, c.msg) {
				t.Errorf("body = %s, want substring %q", body, c.msg)
			}
		})
	}
}

func TestEmbeddingsValidRequestNotImplemented(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	for _, body := range []string{
		`{"model":"m","input":"hello"}`,
		`{"model":"m","input":["a","b"],"encoding_format":"base64"}`,
		`{"model":"m","items":[{"text":"a"},{"image":"http://x/y.png"}]}`,
		`{"model":"m","input":"hi","dimensions":16}`,
	} {
		res := postEmbeddings(t, srv, body)
		if res.StatusCode != http.StatusNotImplemented {
			t.Errorf("body %s: status %d, want 501", body, res.StatusCode)
		}
		res.Body.Close()
	}
}
