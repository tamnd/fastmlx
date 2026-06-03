// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type saveTemplateCase struct {
	Label  string          `json:"label"`
	Error  *string         `json:"error"`
	Record json.RawMessage `json:"record"`
}

type upsertTemplateCase struct {
	Label    string          `json:"label"`
	Existing json.RawMessage `json:"existing"`
	Record   json.RawMessage `json:"record"`
}

type updateTemplateCase struct {
	Label       string          `json:"label"`
	Current     json.RawMessage `json:"current"`
	NewName     *string         `json:"new_name"`
	DisplayName *string         `json:"display_name"`
	Description *string         `json:"description"`
	Settings    *map[string]any `json:"settings"`
	Record      json.RawMessage `json:"record"`
}

type templateFixture struct {
	Now    string               `json:"now"`
	Save   []saveTemplateCase   `json:"save"`
	Upsert []upsertTemplateCase `json:"upsert"`
	Update []updateTemplateCase `json:"update"`
}

func loadTemplateFixture(t *testing.T) templateFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "template_resolve.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx templateFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func TestBuildTemplateRecord(t *testing.T) {
	fx := loadTemplateFixture(t)
	inputs := map[string]struct {
		name, display string
		description   *string
		settings      map[string]any
	}{
		"basic":                         {"creative", "Creative", new("loose sampling"), map[string]any{"temperature": 0.9, "top_p": 0.95}},
		"filters_model_specific":        {"u1", "", nil, map[string]any{"temperature": 0.5, "dflash_enabled": true, "turboquant_kv_bits": float64(3), "is_pinned": true}},
		"display_name_defaults_to_name": {"named", "", nil, map[string]any{}},
		"invalid_name":                  {"Bad Name!", "X", nil, map[string]any{}},
		"empty_settings":                {"empty", "Empty", new("desc"), map[string]any{}},
	}
	for _, tc := range fx.Save {
		t.Run(tc.Label, func(t *testing.T) {
			in, ok := inputs[tc.Label]
			if !ok {
				t.Fatalf("no input for %q", tc.Label)
			}
			rec, err := BuildTemplateRecord(in.name, in.display, in.description, in.settings, fx.Now)
			if tc.Error != nil {
				if err == nil {
					t.Fatalf("want error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildTemplateRecord: %v", err)
			}
			assertRecord(t, rec, tc.Record)
		})
	}
}

func TestUpsertTemplateRecord(t *testing.T) {
	fx := loadTemplateFixture(t)
	inputs := map[string]struct {
		name, display string
		description   *string
		settings      map[string]any
	}{
		"create_new":               {"fast", "Fast", nil, map[string]any{"temperature": 0.2}},
		"replace_keeps_created_at": {"fast", "Faster", new("v2"), map[string]any{"temperature": 0.1, "dflash_enabled": true}},
	}
	for _, tc := range fx.Upsert {
		t.Run(tc.Label, func(t *testing.T) {
			in, ok := inputs[tc.Label]
			if !ok {
				t.Fatalf("no input for %q", tc.Label)
			}
			var existing *GlobalTemplate
			if len(tc.Existing) > 0 && string(tc.Existing) != "null" {
				var g GlobalTemplate
				if err := json.Unmarshal(tc.Existing, &g); err != nil {
					t.Fatalf("decode existing: %v", err)
				}
				existing = &g
			}
			rec, err := UpsertTemplateRecord(in.name, in.display, in.description, in.settings, existing, fx.Now)
			if err != nil {
				t.Fatalf("UpsertTemplateRecord: %v", err)
			}
			assertRecord(t, rec, tc.Record)
		})
	}
}

func TestUpdateTemplateRecord(t *testing.T) {
	fx := loadTemplateFixture(t)
	for _, tc := range fx.Update {
		t.Run(tc.Label, func(t *testing.T) {
			var current GlobalTemplate
			if err := json.Unmarshal(tc.Current, &current); err != nil {
				t.Fatalf("decode current: %v", err)
			}
			rec, err := UpdateTemplateRecord(current, tc.NewName, tc.DisplayName, tc.Description, tc.Settings, fx.Now)
			if err != nil {
				t.Fatalf("UpdateTemplateRecord: %v", err)
			}
			assertRecord(t, rec, tc.Record)
		})
	}
}

func assertRecord(t *testing.T, rec GlobalTemplate, want json.RawMessage) {
	t.Helper()
	out, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal record: %v", err)
	}
	if c := canon(t, out); c != canon(t, want) {
		t.Fatalf("record mismatch\n got: %s\nwant: %s", c, canon(t, want))
	}
}

func BenchmarkUpsertTemplateRecord(b *testing.B) {
	existing := &GlobalTemplate{Name: "fast", DisplayName: "Fast", CreatedAt: "2023-06-01T08:00:00+00:00"}
	settings := map[string]any{"temperature": 0.2, "top_p": 0.9}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = UpsertTemplateRecord("fast", "Fast", nil, settings, existing, "2024-01-15T12:00:00+00:00")
	}
}
