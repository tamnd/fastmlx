// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type pressureCase struct {
	Enforcer map[string]any `json:"enforcer"`
	Out      map[string]any `json:"out"`
}

type memoryCase struct {
	Enforcer map[string]any `json:"enforcer"`
	Status   map[string]any `json:"status"`
	Used     any            `json:"used"`
	Max      any            `json:"max"`
}

type assembleCase struct {
	Models       []any          `json:"models"`
	Status       map[string]any `json:"status"`
	Enforcer     map[string]any `json:"enforcer"`
	TotalActive  int            `json:"total_active"`
	TotalWaiting int            `json:"total_waiting"`
	Out          map[string]any `json:"out"`
}

type activeModelsFixture struct {
	Pressure []pressureCase `json:"pressure"`
	Memory   []memoryCase   `json:"memory"`
	Assemble []assembleCase `json:"assemble"`
	Empty    map[string]any `json:"empty"`
}

func loadActiveModels(t *testing.T) activeModelsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/activemodels.json")
	if err != nil {
		t.Fatal(err)
	}
	var f activeModelsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBuildMemoryPressureParity(t *testing.T) {
	for i, c := range loadActiveModels(t).Pressure {
		got := jsonRoundTrip(t, BuildMemoryPressure(c.Enforcer))
		want := jsonRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("BuildMemoryPressure case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestSelectModelMemoryParity(t *testing.T) {
	for i, c := range loadActiveModels(t).Memory {
		used, mx := SelectModelMemory(c.Enforcer, c.Status)
		got := jsonRoundTrip(t, map[string]any{"used": used, "max": mx})
		want := jsonRoundTrip(t, map[string]any{"used": c.Used, "max": c.Max})
		if !reflect.DeepEqual(got, want) {
			t.Errorf("SelectModelMemory case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestAssembleActiveModelsDataParity(t *testing.T) {
	for i, c := range loadActiveModels(t).Assemble {
		got := jsonRoundTrip(t, AssembleActiveModelsData(c.Models, c.Status, c.Enforcer, c.TotalActive, c.TotalWaiting))
		want := jsonRoundTrip(t, c.Out)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("AssembleActiveModelsData case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func TestEmptyActiveModelsDataParity(t *testing.T) {
	got := jsonRoundTrip(t, EmptyActiveModelsData())
	want := jsonRoundTrip(t, loadActiveModels(t).Empty)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EmptyActiveModelsData:\n got  %v\n want %v", got, want)
	}
}

func BenchmarkBuildMemoryPressure(b *testing.B) {
	es := map[string]any{"enabled": true, "current_bytes": 1000, "soft_bytes": 2000,
		"hard_bytes": 3000, "current_formatted": "1.0GB", "soft_formatted": "2.0GB",
		"hard_formatted": "3.0GB", "pressure_level": "warn"}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildMemoryPressure(es)
	}
}

func BenchmarkAssembleActiveModelsData(b *testing.B) {
	models := []any{map[string]any{"id": "m1"}}
	status := map[string]any{"current_model_memory": 111, "final_ceiling": 222}
	b.ReportAllocs()
	for b.Loop() {
		_ = AssembleActiveModelsData(models, status, nil, 3, 1)
	}
}
