// SPDX-License-Identifier: MIT OR Apache-2.0

package server

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/tamnd/fastmlx/api"
)

func encodeJSON(w io.Writer, v any) error {
	return json.NewEncoder(w).Encode(v)
}

// publicPaths bypass authentication (liveness/readiness probes).
var publicPaths = map[string]bool{
	"/health":     true,
	"/api/status": true,
}

// withAuth enforces API-key auth when keys are configured. It accepts
// Authorization: Bearer <key> or x-api-key: <key> (Anthropic SDK), matched
// constant-time against any configured key.
func (a *App) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(a.cfg.APIKeys) == 0 || publicPaths[r.URL.Path] || r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		if a.keyOK(presentedKey(r)) {
			next.ServeHTTP(w, r)
			return
		}
		writeError(w, http.StatusUnauthorized, "invalid or missing API key", "authentication_error")
	})
}

func presentedKey(r *http.Request) string {
	if h := r.Header.Get("Authorization"); h != "" {
		if rest, ok := strings.CutPrefix(h, "Bearer "); ok {
			return strings.TrimSpace(rest)
		}
	}
	return strings.TrimSpace(r.Header.Get("x-api-key"))
}

func (a *App) keyOK(presented string) bool {
	if presented == "" {
		return false
	}
	ok := false
	for _, k := range a.cfg.APIKeys {
		if subtle.ConstantTimeCompare([]byte(presented), []byte(k)) == 1 {
			ok = true
		}
	}
	return ok
}

// withCORS applies CORS headers and answers preflight. With a "*" origin it
// echoes the request Origin and allows credentials (a bare "*" is invalid with
// credentials), a permissive default.
func (a *App) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && a.originAllowed(origin) {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", origin)
			h.Set("Vary", "Origin")
			h.Set("Access-Control-Allow-Credentials", "true")
			h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (a *App) originAllowed(origin string) bool {
	for _, o := range a.cfg.CORSOrigins {
		if o == "*" || o == origin {
			return true
		}
	}
	return false
}

// withRecover turns a handler panic into a sanitized 500 instead of dropping the
// connection.
func (a *App) withRecover(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				a.logger.Printf("panic serving %s %s: %v", r.Method, r.URL.Path, rec)
				writeError(w, http.StatusInternalServerError, "internal server error", "internal_error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// withRequestLog injects a request id and logs each request's outcome.
func (a *App) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rid := r.Header.Get("X-Request-ID")
		if rid == "" {
			rid = newRequestID()
		}
		w.Header().Set("X-Request-ID", rid)
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		start := time.Now()
		next.ServeHTTP(sw, r)
		// Filter the admin stats polling noise from the access log.
		if !strings.HasPrefix(r.URL.Path, "/admin/api/stats") {
			a.logger.Printf("%s %s %d %s rid=%s", r.Method, r.URL.Path, sw.status, time.Since(start).Round(time.Millisecond), rid)
		}
	})
}

// statusWriter captures the response status and preserves http.Flusher so SSE
// streaming keeps working through the middleware chain.
type statusWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (s *statusWriter) WriteHeader(code int) {
	if !s.wroteHeader {
		s.status = code
		s.wroteHeader = true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusWriter) Write(b []byte) (int, error) {
	s.wroteHeader = true
	return s.ResponseWriter.Write(b)
}

func (s *statusWriter) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// writeError emits the OpenAI error envelope.
func writeError(w http.ResponseWriter, status int, message, typ string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = encodeJSON(w, api.NewError(message, typ, ""))
}
