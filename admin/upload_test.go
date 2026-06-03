// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type uploadNameCase struct {
	Name string `json:"name"`
	Out  bool   `json:"out"`
}

type uploadReadmeCase struct {
	Text string `json:"text"`
	Out  bool   `json:"out"`
}

type uploadTaskCase struct {
	Task struct {
		TaskID      string  `json:"task_id"`
		ModelName   string  `json:"model_name"`
		ModelPath   string  `json:"model_path"`
		RepoID      string  `json:"repo_id"`
		Status      string  `json:"status"`
		Progress    float64 `json:"progress"`
		Error       string  `json:"error"`
		CreatedAt   float64 `json:"created_at"`
		StartedAt   float64 `json:"started_at"`
		CompletedAt float64 `json:"completed_at"`
		TotalSize   int     `json:"total_size"`
		RepoURL     string  `json:"repo_url"`
	} `json:"task"`
	Out map[string]any `json:"out"`
}

type uploadFixture struct {
	Names   []uploadNameCase   `json:"names"`
	Readmes []uploadReadmeCase `json:"readmes"`
	Tasks   []uploadTaskCase   `json:"tasks"`
}

func loadUpload(t *testing.T) uploadFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/upload.json")
	if err != nil {
		t.Fatal(err)
	}
	var f uploadFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestIsOQModelParity(t *testing.T) {
	for i, c := range loadUpload(t).Names {
		if got := IsOQModel(c.Name); got != c.Out {
			t.Errorf("IsOQModel case %d (%q) = %v, want %v", i, c.Name, got, c.Out)
		}
	}
}

func TestHasMeaningfulReadmeParity(t *testing.T) {
	for i, c := range loadUpload(t).Readmes {
		if got := HasMeaningfulReadme(c.Text); got != c.Out {
			t.Errorf("HasMeaningfulReadme case %d (%q) = %v, want %v", i, c.Text, got, c.Out)
		}
	}
}

func TestUploadTaskToDictParity(t *testing.T) {
	for i, c := range loadUpload(t).Tasks {
		task := UploadTask{
			TaskID:      c.Task.TaskID,
			ModelName:   c.Task.ModelName,
			ModelPath:   c.Task.ModelPath,
			RepoID:      c.Task.RepoID,
			Status:      c.Task.Status,
			Progress:    c.Task.Progress,
			Error:       c.Task.Error,
			CreatedAt:   c.Task.CreatedAt,
			StartedAt:   c.Task.StartedAt,
			CompletedAt: c.Task.CompletedAt,
			TotalSize:   c.Task.TotalSize,
			RepoURL:     c.Task.RepoURL,
		}
		got := jsonRoundTrip(t, task.ToDict())
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("UploadTask.ToDict case %d:\n got  %v\n want %v", i, got, c.Out)
		}
	}
}

func BenchmarkHasMeaningfulReadme(b *testing.B) {
	text := "---\nlibrary_name: mlx\n---\n\n# Title\nBody text here."
	b.ReportAllocs()
	for b.Loop() {
		_ = HasMeaningfulReadme(text)
	}
}
