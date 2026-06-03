// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type applyCase struct {
	Label           string          `json:"label"`
	Current         json.RawMessage `json:"current"`
	ProfileSettings map[string]any  `json:"profile_settings"`
	Name            string          `json:"name"`
	Result          json.RawMessage `json:"result"`
}

type recordCase struct {
	Label  string          `json:"label"`
	Error  *string         `json:"error"`
	Record json.RawMessage `json:"record"`
}

type resolveFixture struct {
	Now    string       `json:"now"`
	Apply  []applyCase  `json:"apply"`
	Record []recordCase `json:"record"`
}

func loadResolveFixture(t *testing.T) resolveFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "profile_resolve.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx resolveFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

func TestApplyProfileToSettings(t *testing.T) {
	fx := loadResolveFixture(t)
	for _, tc := range fx.Apply {
		t.Run(tc.Label, func(t *testing.T) {
			var current *ModelSettings
			// A JSON null current means start from defaults.
			if len(tc.Current) > 0 && string(tc.Current) != "null" {
				m := NewModelSettings()
				if err := json.Unmarshal(tc.Current, &m); err != nil {
					t.Fatalf("decode current: %v", err)
				}
				current = &m
			}
			got, err := ApplyProfileToSettings(current, tc.ProfileSettings, tc.Name)
			if err != nil {
				t.Fatalf("ApplyProfileToSettings: %v", err)
			}
			out, err := got.ToDict()
			if err != nil {
				t.Fatalf("ToDict: %v", err)
			}
			if c := canon(t, out); c != canon(t, tc.Result) {
				t.Fatalf("result mismatch\n got: %s\nwant: %s", c, canon(t, tc.Result))
			}
		})
	}
}

func TestBuildProfileRecord(t *testing.T) {
	fx := loadResolveFixture(t)
	inputs := map[string]struct {
		name, display  string
		description    *string
		settings       map[string]any
		sourceTemplate *string
	}{
		"basic":                         {"fast", "Fast", new("quick replies"), map[string]any{"temperature": 0.3, "top_p": 0.9}, nil},
		"filters_non_profile_keys":      {"p1", "", nil, map[string]any{"temperature": 0.5, "is_pinned": true, "display_name": "x", "bogus": float64(1), "turboquant_kv_bits": float64(3)}, new("tmpl-a")},
		"display_name_defaults_to_name": {"named", "", nil, map[string]any{}, nil},
		"invalid_name":                  {"Bad Name!", "X", nil, map[string]any{}, nil},
		"empty_settings":                {"empty", "Empty", new("desc"), map[string]any{}, nil},
	}
	for _, tc := range fx.Record {
		t.Run(tc.Label, func(t *testing.T) {
			in, ok := inputs[tc.Label]
			if !ok {
				t.Fatalf("no input for %q", tc.Label)
			}
			rec, err := BuildProfileRecord(in.name, in.display, in.description, in.settings, in.sourceTemplate, fx.Now)
			if tc.Error != nil {
				if err == nil {
					t.Fatalf("want error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildProfileRecord: %v", err)
			}
			out, err := json.Marshal(rec)
			if err != nil {
				t.Fatalf("marshal record: %v", err)
			}
			if c := canon(t, out); c != canon(t, tc.Record) {
				t.Fatalf("record mismatch\n got: %s\nwant: %s", c, canon(t, tc.Record))
			}
		})
	}
}

func BenchmarkApplyProfileToSettings(b *testing.B) {
	cur := NewModelSettings()
	ps := map[string]any{"temperature": 0.2, "top_k": float64(50)}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ApplyProfileToSettings(&cur, ps, "p")
	}
}
