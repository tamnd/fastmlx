// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type embeddingFixture struct {
	Responses []struct {
		Model          string      `json:"model"`
		Embeddings     [][]float64 `json:"embeddings"`
		TotalTokens    int         `json:"total_tokens"`
		EncodingFormat string      `json:"encoding_format"`
		Dimensions     *int        `json:"dimensions"`
		Expected       string      `json:"expected"`
	} `json:"responses"`
	Base64 []struct {
		Input    []float64 `json:"input"`
		Expected string    `json:"expected"`
	} `json:"base64"`
	Truncate []struct {
		Input      []float64 `json:"input"`
		Dimensions int       `json:"dimensions"`
		Expected   []float64 `json:"expected"`
	} `json:"truncate"`
}

func loadEmbeddingFixture(t *testing.T) embeddingFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/embedding.json")
	if err != nil {
		t.Fatal(err)
	}
	var f embeddingFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBuildEmbeddingResponseParity(t *testing.T) {
	fx := loadEmbeddingFixture(t)
	for i, c := range fx.Responses {
		got := BuildEmbeddingResponse(c.Model, c.Embeddings, c.TotalTokens, c.EncodingFormat, c.Dimensions)
		if got != c.Expected {
			t.Errorf("case %d (%s):\n got  %s\n want %s", i, c.Model, got, c.Expected)
		}
	}
}

func TestEncodeEmbeddingBase64Parity(t *testing.T) {
	fx := loadEmbeddingFixture(t)
	for i, c := range fx.Base64 {
		if got := EncodeEmbeddingBase64(c.Input); got != c.Expected {
			t.Errorf("case %d: got %q want %q", i, got, c.Expected)
		}
	}
}

func TestTruncateEmbeddingParity(t *testing.T) {
	fx := loadEmbeddingFixture(t)
	for i, c := range fx.Truncate {
		got := TruncateEmbedding(c.Input, c.Dimensions)
		if len(got) != len(c.Expected) {
			t.Fatalf("case %d: len %d want %d", i, len(got), len(c.Expected))
		}
		for j := range got {
			if got[j] != c.Expected[j] {
				t.Errorf("case %d elem %d: got %v want %v", i, j, got[j], c.Expected[j])
			}
		}
	}
}

func TestTruncateEmbeddingNoTruncationWhenWideEnough(t *testing.T) {
	in := []float64{0.1, 0.2, 0.3}
	if got := TruncateEmbedding(in, 5); len(got) != 3 {
		t.Errorf("asking for more dims than present should return the vector unchanged, got len %d", len(got))
	}
}

func BenchmarkBuildEmbeddingResponse(b *testing.B) {
	emb := make([]float64, 384)
	for i := range emb {
		emb[i] = float64(i%7) * 0.1
	}
	embeddings := [][]float64{emb, emb}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildEmbeddingResponse("m", embeddings, 12, "float", nil)
	}
}
