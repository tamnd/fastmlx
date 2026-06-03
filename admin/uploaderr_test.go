// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type uploadErrFixture struct {
	Cases []struct {
		CFMitigated string `json:"cf_mitigated"`
		Body        string `json:"body"`
		Status      int    `json:"status"`
		Out         string `json:"out"`
	} `json:"cases"`
}

func loadUploadErr(t *testing.T) uploadErrFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/uploaderr.json")
	if err != nil {
		t.Fatal(err)
	}
	var f uploadErrFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSanitizeUploadErrorParity(t *testing.T) {
	for i, c := range loadUploadErr(t).Cases {
		got := SanitizeUploadError(UploadResponse{
			CFMitigated: c.CFMitigated,
			Body:        c.Body,
			StatusCode:  c.Status,
			HasStatus:   true,
		})
		if got != c.Out {
			t.Errorf("SanitizeUploadError case %d:\n got  %q\n want %q", i, got, c.Out)
		}
	}
}

func TestSanitizeUploadErrorNoStatus(t *testing.T) {
	// With no status code the bare-status fallback reads "HTTP ?", matching
	// getattr(resp, "status_code", "?") on a response without one.
	got := SanitizeUploadError(UploadResponse{Body: "", HasStatus: false})
	if got != "HTTP ?" {
		t.Errorf("SanitizeUploadError no-status = %q, want %q", got, "HTTP ?")
	}
}

func BenchmarkSanitizeUploadError(b *testing.B) {
	r := UploadResponse{Body: `{"error":"rate limit exceeded"}`, StatusCode: 429, HasStatus: true}
	b.ReportAllocs()
	for b.Loop() {
		_ = SanitizeUploadError(r)
	}
}
