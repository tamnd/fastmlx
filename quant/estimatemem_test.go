// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type estimateMemCase struct {
	Size int            `json:"size"`
	Out  map[string]any `json:"out"`
}

type estimateMemFixture struct {
	Mem []estimateMemCase `json:"mem"`
}

func loadEstimateMem(t *testing.T) estimateMemFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/estimatemem.json")
	if err != nil {
		t.Fatal(err)
	}
	var f estimateMemFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestEstimateMemoryParity(t *testing.T) {
	for i, c := range loadEstimateMem(t).Mem {
		b, err := json.Marshal(EstimateMemory(c.Size))
		if err != nil {
			t.Fatal(err)
		}
		var got map[string]any
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("EstimateMemory case %d (%d):\n got  %v\n want %v", i, c.Size, got, c.Out)
		}
	}
}

func BenchmarkEstimateMemory(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = EstimateMemory(8589934592)
	}
}
