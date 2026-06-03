// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type posSensFixture struct {
	Cases []struct {
		NumLayers int                `json:"num_layers"`
		Out       map[string]float64 `json:"out"`
	} `json:"cases"`
}

func loadPosSens(t *testing.T) posSensFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/possens.json")
	if err != nil {
		t.Fatal(err)
	}
	var f posSensFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestPositionSensitivityMapParity(t *testing.T) {
	for i, c := range loadPosSens(t).Cases {
		got := PositionSensitivityMap(c.NumLayers)
		want := c.Out
		if want == nil {
			want = map[string]float64{}
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("PositionSensitivityMap case %d (n=%d):\n got  %v\n want %v", i, c.NumLayers, got, want)
		}
	}
}

func BenchmarkPositionSensitivityMap(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = PositionSensitivityMap(80)
	}
}
