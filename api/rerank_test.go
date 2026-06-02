// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

// Fixtures in testdata/parity/rerank.json are captured from the reference
// RerankResponse pydantic model serialized the way FastAPI renders it
// (json.dumps with compact separators and ensure_ascii=False). The body is a
// contractual HTTP response, so the comparison is byte-exact: the port builds
// the same object with the same sentinel id and serializes with dumpCompact.

type rerankCase struct {
	Label string `json:"label"`
	Input struct {
		Documents       []json.RawMessage `json:"documents"`
		Scores          []float64         `json:"scores"`
		Order           []int             `json:"order"`
		ReturnDocuments bool              `json:"return_documents"`
		TotalTokens     int               `json:"total_tokens"`
		Model           string            `json:"model"`
		ID              string            `json:"id"`
	} `json:"input"`
	Raw string `json:"raw"`
}

func loadRerankCases(t testing.TB) []rerankCase {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/rerank.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx struct {
		Cases []rerankCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("decode fixtures: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("no fixture cases")
	}
	return fx.Cases
}

func TestBuildRerankResponseParity(t *testing.T) {
	for _, c := range loadRerankCases(t) {
		t.Run(c.Label, func(t *testing.T) {
			got := BuildRerankResponse(c.Input.ID, c.Input.Model, c.Input.Scores,
				c.Input.Order, c.Input.Documents, c.Input.ReturnDocuments, c.Input.TotalTokens)
			if got != c.Raw {
				t.Errorf("mismatch\n got: %s\nwant: %s", got, c.Raw)
			}
		})
	}
}

func TestNormalizeDocuments(t *testing.T) {
	docs := []json.RawMessage{
		json.RawMessage(`"plain string"`),
		json.RawMessage(`{"text":"from text field","image":"x"}`),
		json.RawMessage(`{"image":"only"}`),
	}
	got := NormalizeDocuments(docs)
	want := []string{"plain string", "from text field", ""}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestNewRerankID(t *testing.T) {
	id := NewRerankID()
	if len(id) != len("rerank-")+8 {
		t.Fatalf("id %q has wrong length", id)
	}
	if id[:7] != "rerank-" {
		t.Errorf("id %q missing rerank- prefix", id)
	}
}

func TestFormatPyFloat(t *testing.T) {
	cases := map[float64]string{
		0.0:   "0.0",
		1.0:   "1.0",
		0.95:  "0.95",
		0.1:   "0.1",
		0.78:  "0.78",
		0.123: "0.123",
	}
	for f, want := range cases {
		if got := formatPyFloat(f); got != want {
			t.Errorf("formatPyFloat(%v) = %q, want %q", f, got, want)
		}
	}
}

func BenchmarkBuildRerankResponse(b *testing.B) {
	cases := loadRerankCases(b)
	c := cases[0]
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildRerankResponse(c.Input.ID, c.Input.Model, c.Input.Scores,
			c.Input.Order, c.Input.Documents, c.Input.ReturnDocuments, c.Input.TotalTokens)
	}
}
