// SPDX-License-Identifier: MIT OR Apache-2.0

package server

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// These exercise the /v1/audio/* routes' request validation against the mock
// backend. The transcription, speech, and processing computation is served by
// the STT/TTS/STS engines (compute backend, later milestone), so a valid
// request reports not-implemented; the validation branches mirror the reference.

func postSpeech(t *testing.T, srv *httptest.Server, body string) *http.Response {
	t.Helper()
	res, err := http.Post(srv.URL+"/v1/audio/speech", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// postAudioUpload builds a multipart request. A file part is included unless
// withFile is false; a model field is included unless model is empty.
func postAudioUpload(t *testing.T, srv *httptest.Server, path string, withFile bool, model string) *http.Response {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if withFile {
		fw, err := mw.CreateFormFile("file", "audio.wav")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fw.Write([]byte("RIFFmock-audio-bytes"))
	}
	if model != "" {
		_ = mw.WriteField("model", model)
	}
	mw.Close()
	res, err := http.Post(srv.URL+path, mw.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func bodyString(t *testing.T, res *http.Response) string {
	t.Helper()
	buf := make([]byte, 4096)
	n, _ := res.Body.Read(buf)
	return string(buf[:n])
}

func TestSpeechValidation(t *testing.T) {
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
		{"empty input", `{"model":"m","input":""}`, 400, "must not be empty"},
		{"whitespace input", `{"model":"m","input":"   "}`, 400, "must not be empty"},
		{"streaming non-wav format", `{"model":"m","input":"hi","stream":true,"response_format":"mp3"}`, 400, "only supports response_format='wav'"},
		{"streaming interval too small", `{"model":"m","input":"hi","stream":true,"streaming_interval":0.001}`, 400, "at least 0.01 seconds"},
		{"ref_audio without ref_text", `{"model":"m","input":"hi","ref_audio":"aGVsbG8="}`, 400, "ref_text"},
		{"ref_audio bad base64", `{"model":"m","input":"hi","ref_audio":"not base64!!","ref_text":"x"}`, 400, "Invalid base64"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := postSpeech(t, srv, c.body)
			defer res.Body.Close()
			body := bodyString(t, res)
			if res.StatusCode != c.code {
				t.Fatalf("status %d, want %d: %s", res.StatusCode, c.code, body)
			}
			if !strings.Contains(body, c.msg) {
				t.Errorf("body = %s, want substring %q", body, c.msg)
			}
		})
	}
}

func TestSpeechValidRequestNotImplemented(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	for _, body := range []string{
		`{"model":"m","input":"hello world"}`,
		`{"model":"m","input":"hi","stream":true,"response_format":"wav","streaming_interval":0.2}`,
		`{"model":"m","input":"hi","ref_audio":"aGVsbG8=","ref_text":"hello"}`,
	} {
		res := postSpeech(t, srv, body)
		if res.StatusCode != http.StatusNotImplemented {
			t.Errorf("body %s: status %d, want 501", body, res.StatusCode)
		}
		res.Body.Close()
	}
}

func TestAudioUploadValidation(t *testing.T) {
	app, stop := newTestApp(t, "x", nil)
	defer stop()
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	for _, path := range []string{"/v1/audio/transcriptions", "/v1/audio/process"} {
		t.Run(path+" missing file", func(t *testing.T) {
			res := postAudioUpload(t, srv, path, false, "m")
			defer res.Body.Close()
			if res.StatusCode != http.StatusUnprocessableEntity {
				t.Fatalf("status %d, want 422: %s", res.StatusCode, bodyString(t, res))
			}
		})
		t.Run(path+" missing model", func(t *testing.T) {
			res := postAudioUpload(t, srv, path, true, "")
			defer res.Body.Close()
			if res.StatusCode != http.StatusUnprocessableEntity {
				t.Fatalf("status %d, want 422: %s", res.StatusCode, bodyString(t, res))
			}
		})
		t.Run(path+" valid request not implemented", func(t *testing.T) {
			res := postAudioUpload(t, srv, path, true, "m")
			defer res.Body.Close()
			if res.StatusCode != http.StatusNotImplemented {
				t.Fatalf("status %d, want 501: %s", res.StatusCode, bodyString(t, res))
			}
		})
	}
}
