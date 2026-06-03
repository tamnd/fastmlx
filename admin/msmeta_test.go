// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type msFilesCase struct {
	Files []map[string]any `json:"files"`
	Out   int              `json:"out"`
}

type msParamsCase struct {
	Config map[string]any `json:"config"`
	Out    int            `json:"out"`
}

type msEntryCase struct {
	Entry map[string]any `json:"entry"`
	Out   map[string]any `json:"out"`
}

type msMetaFixture struct {
	Files   []msFilesCase  `json:"files"`
	Params  []msParamsCase `json:"params"`
	Entries []msEntryCase  `json:"entries"`
}

func loadMSMeta(t *testing.T) msMetaFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/msmeta.json")
	if err != nil {
		t.Fatal(err)
	}
	var f msMetaFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestExtractModelSizeFromFilesParity(t *testing.T) {
	for i, c := range loadMSMeta(t).Files {
		if got := ExtractModelSizeFromFiles(c.Files); got != c.Out {
			t.Errorf("ExtractModelSizeFromFiles case %d (%v) = %d, want %d", i, c.Files, got, c.Out)
		}
	}
}

func TestEstimateParamsFromConfigParity(t *testing.T) {
	for i, c := range loadMSMeta(t).Params {
		if got := EstimateParamsFromConfig(c.Config); got != c.Out {
			t.Errorf("EstimateParamsFromConfig case %d (%v) = %d, want %d", i, c.Config, got, c.Out)
		}
	}
}

func TestParseMSModelEntryParity(t *testing.T) {
	for i, c := range loadMSMeta(t).Entries {
		got := jsonRoundTrip(t, ParseMSModelEntry(c.Entry))
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("ParseMSModelEntry case %d (%v):\n got  %v\n want %v", i, c.Entry, got, c.Out)
		}
	}
}

func BenchmarkEstimateParamsFromConfig(b *testing.B) {
	config := map[string]any{
		"vocab_size": 151936.0, "hidden_size": 2048.0, "num_hidden_layers": 36.0,
		"intermediate_size": 11008.0, "num_attention_heads": 16.0,
		"num_key_value_heads": 2.0, "head_dim": 128.0, "tie_word_embeddings": true,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = EstimateParamsFromConfig(config)
	}
}

func BenchmarkParseMSModelEntry(b *testing.B) {
	entry := map[string]any{
		"Path": "qwen", "Name": "Qwen3-4B", "Downloads": 5000.0,
		"Likes": 42.0, "StorageSize": 8589934592.0,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = ParseMSModelEntry(entry)
	}
}
