// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type clampCase struct {
	N   int `json:"n"`
	Out int `json:"out"`
}

type logNameCase struct {
	Name string `json:"name"`
	Out  bool   `json:"out"`
}

type unsafeFileCase struct {
	File string `json:"file"`
	Out  bool   `json:"out"`
}

type tailCase struct {
	Content string `json:"content"`
	N       int    `json:"n"`
	Out     string `json:"out"`
	Total   int    `json:"total"`
}

type logsFixture struct {
	Clamps []clampCase      `json:"clamps"`
	Names  []logNameCase    `json:"names"`
	Files  []unsafeFileCase `json:"files"`
	Tails  []tailCase       `json:"tails"`
}

func loadLogs(t *testing.T) logsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/logs.json")
	if err != nil {
		t.Fatal(err)
	}
	var f logsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestClampLogLinesParity(t *testing.T) {
	for i, c := range loadLogs(t).Clamps {
		if got := ClampLogLines(c.N); got != c.Out {
			t.Errorf("ClampLogLines case %d (%d) = %d, want %d", i, c.N, got, c.Out)
		}
	}
}

func TestIsLogFileNameParity(t *testing.T) {
	for i, c := range loadLogs(t).Names {
		if got := IsLogFileName(c.Name); got != c.Out {
			t.Errorf("IsLogFileName case %d (%q) = %v, want %v", i, c.Name, got, c.Out)
		}
	}
}

func TestIsUnsafeLogFileParity(t *testing.T) {
	for i, c := range loadLogs(t).Files {
		if got := IsUnsafeLogFile(c.File); got != c.Out {
			t.Errorf("IsUnsafeLogFile case %d (%q) = %v, want %v", i, c.File, got, c.Out)
		}
	}
}

func TestTailContentParity(t *testing.T) {
	for i, c := range loadLogs(t).Tails {
		out, total := TailContent(c.Content, c.N)
		if out != c.Out || total != c.Total {
			t.Errorf("TailContent case %d (%q, n=%d) = (%q, %d), want (%q, %d)",
				i, c.Content, c.N, out, total, c.Out, c.Total)
		}
	}
}

func BenchmarkTailContent(b *testing.B) {
	content := "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n"
	b.ReportAllocs()
	for b.Loop() {
		_, _ = TailContent(content, 3)
	}
}

func BenchmarkIsLogFileName(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsLogFileName("server.log.2024-01-01")
	}
}
