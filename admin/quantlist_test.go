// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type mtpCase struct {
	Config map[string]any `json:"config"`
	Out    bool           `json:"out"`
}

type quantInfoCase struct {
	Config map[string]any `json:"config"`
	Name   string         `json:"name"`
	Path   string         `json:"path"`
	Size   int            `json:"size"`
	HasMTP bool           `json:"has_mtp"`
	Info   map[string]any `json:"info"`
	Source map[string]any `json:"source"`
}

type quantListFixture struct {
	MTP  []mtpCase       `json:"mtp"`
	Info []quantInfoCase `json:"info"`
}

func loadQuantList(t *testing.T) quantListFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/quantlist.json")
	if err != nil {
		t.Fatal(err)
	}
	var f quantListFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestConfigDeclaresMTPParity(t *testing.T) {
	for i, c := range loadQuantList(t).MTP {
		if got := ConfigDeclaresMTP(c.Config); got != c.Out {
			t.Errorf("ConfigDeclaresMTP case %d (%v) = %v, want %v", i, c.Config, got, c.Out)
		}
	}
}

func TestQuantizableModelInfoParity(t *testing.T) {
	for i, c := range loadQuantList(t).Info {
		gotInfo := jsonRoundTrip(t, QuantizableModelInfo(c.Config, c.Name, c.Path, c.Size, c.HasMTP))
		if !reflect.DeepEqual(gotInfo, c.Info) {
			t.Errorf("QuantizableModelInfo case %d:\n got  %v\n want %v", i, gotInfo, c.Info)
		}
		gotSource := jsonRoundTrip(t, SourceModelInfo(c.Config, c.Name, c.Path, c.Size, c.HasMTP))
		if !reflect.DeepEqual(gotSource, c.Source) {
			t.Errorf("SourceModelInfo case %d:\n got  %v\n want %v", i, gotSource, c.Source)
		}
	}
}

func BenchmarkSourceModelInfo(b *testing.B) {
	config := map[string]any{
		"model_type": "qwen3", "num_hidden_layers": 36.0, "num_local_experts": 0.0,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = SourceModelInfo(config, "Qwen3-4B", "/m/Qwen3-4B", 8589934592, false)
	}
}
