// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type oqSizeCase struct {
	Bytes int    `json:"bytes"`
	Out   string `json:"out"`
}

type oqPhaseCase struct {
	Phase   string  `json:"phase"`
	OQLevel float64 `json:"oq_level"`
	Out     string  `json:"out"`
}

type oqTaskCase struct {
	Task struct {
		TaskID      string  `json:"task_id"`
		ModelName   string  `json:"model_name"`
		ModelPath   string  `json:"model_path"`
		OQLevel     float64 `json:"oq_level"`
		OutputName  string  `json:"output_name"`
		OutputPath  string  `json:"output_path"`
		Status      string  `json:"status"`
		Progress    float64 `json:"progress"`
		Phase       string  `json:"phase"`
		Error       string  `json:"error"`
		CreatedAt   float64 `json:"created_at"`
		StartedAt   float64 `json:"started_at"`
		CompletedAt float64 `json:"completed_at"`
		SourceSize  int     `json:"source_size"`
		OutputSize  int     `json:"output_size"`
		Dtype       string  `json:"dtype"`
	} `json:"task"`
	Out map[string]any `json:"out"`
}

type oqTaskFixture struct {
	Sizes  []oqSizeCase  `json:"sizes"`
	Phases []oqPhaseCase `json:"phases"`
	Tasks  []oqTaskCase  `json:"tasks"`
}

func loadOQTask(t *testing.T) oqTaskFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/oqtask.json")
	if err != nil {
		t.Fatal(err)
	}
	var f oqTaskFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFormatSizeParity(t *testing.T) {
	for i, c := range loadOQTask(t).Sizes {
		if got := FormatSize(c.Bytes); got != c.Out {
			t.Errorf("FormatSize case %d (%d) = %q, want %q", i, c.Bytes, got, c.Out)
		}
	}
}

func TestPhaseLabelParity(t *testing.T) {
	for i, c := range loadOQTask(t).Phases {
		if got := PhaseLabel(c.Phase, c.OQLevel); got != c.Out {
			t.Errorf("PhaseLabel case %d (%q, %g) = %q, want %q", i, c.Phase, c.OQLevel, got, c.Out)
		}
	}
}

func TestQuantTaskToDictParity(t *testing.T) {
	for i, c := range loadOQTask(t).Tasks {
		task := QuantTask{
			TaskID:      c.Task.TaskID,
			ModelName:   c.Task.ModelName,
			ModelPath:   c.Task.ModelPath,
			OQLevel:     c.Task.OQLevel,
			OutputName:  c.Task.OutputName,
			OutputPath:  c.Task.OutputPath,
			Status:      c.Task.Status,
			Progress:    c.Task.Progress,
			Phase:       c.Task.Phase,
			Error:       c.Task.Error,
			CreatedAt:   c.Task.CreatedAt,
			StartedAt:   c.Task.StartedAt,
			CompletedAt: c.Task.CompletedAt,
			SourceSize:  c.Task.SourceSize,
			OutputSize:  c.Task.OutputSize,
			Dtype:       c.Task.Dtype,
		}
		got := jsonRoundTrip(t, task.ToDict())
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("QuantTask.ToDict case %d:\n got  %v\n want %v", i, got, c.Out)
		}
	}
}

func BenchmarkPhaseLabel(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = PhaseLabel("quantizing_eta|792|879|0:02", 2.5)
	}
}

func BenchmarkFormatSize(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = FormatSize(7516192768)
	}
}
