// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type profilesFixture struct {
	UniversalFields     []string `json:"universal_fields"`
	ModelSpecificFields []string `json:"model_specific_fields"`
	ExcludedFields      []string `json:"excluded_fields"`
	Filter              struct {
		Universal json.RawMessage `json:"universal"`
		Profile   json.RawMessage `json:"profile"`
	} `json:"filter"`
	ProfileRoundtrip  json.RawMessage `json:"profile_roundtrip"`
	TemplateRoundtrip json.RawMessage `json:"template_roundtrip"`
	MinimalRoundtrip  json.RawMessage `json:"minimal_roundtrip"`
	Validations       []struct {
		Name  string `json:"name"`
		Valid bool   `json:"valid"`
	} `json:"validations"`
}

func loadProfilesFixture(t *testing.T) profilesFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/model_profiles.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f profilesFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return f
}

// canon re-marshals a JSON value so map keys sort consistently for comparison.
func canon(t *testing.T, raw []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("canon decode: %v", err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("canon encode: %v", err)
	}
	return string(out)
}

func marshalCanon(t *testing.T, v any) string {
	t.Helper()
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return canon(t, out)
}

// sampleSettings mirrors the settings dict used in the capture.
func sampleSettings() map[string]any {
	return map[string]any{
		"max_tokens":           512,
		"temperature":          0.5,
		"top_k":                40,
		"chat_template_kwargs": map[string]any{"enable_thinking": true, "nested": map[string]any{"x": 1}},
		"dflash_enabled":       true,
		"specprefill_keep_pct": 0.25,
		"is_pinned":            true,
		"trust_remote_code":    false,
		"totally_unknown":      123,
	}
}

func TestFieldLists(t *testing.T) {
	f := loadProfilesFixture(t)
	if !reflect.DeepEqual(UniversalProfileFields, f.UniversalFields) {
		t.Errorf("universal fields mismatch:\n got %v\nwant %v", UniversalProfileFields, f.UniversalFields)
	}
	if !reflect.DeepEqual(ModelSpecificProfileFields, f.ModelSpecificFields) {
		t.Errorf("model-specific fields mismatch:\n got %v\nwant %v", ModelSpecificProfileFields, f.ModelSpecificFields)
	}
	got := make([]string, 0, len(ExcludedFromProfiles))
	for k := range ExcludedFromProfiles {
		got = append(got, k)
	}
	want := map[string]bool{}
	for _, k := range f.ExcludedFields {
		want[k] = true
	}
	gotSet := map[string]bool{}
	for _, k := range got {
		gotSet[k] = true
	}
	if !reflect.DeepEqual(gotSet, want) {
		t.Errorf("excluded fields mismatch:\n got %v\nwant %v", gotSet, want)
	}
}

func TestFilterFields(t *testing.T) {
	f := loadProfilesFixture(t)
	if got, want := marshalCanon(t, FilterUniversalFields(sampleSettings())), canon(t, f.Filter.Universal); got != want {
		t.Errorf("FilterUniversalFields:\n got %s\nwant %s", got, want)
	}
	if got, want := marshalCanon(t, FilterProfileFields(sampleSettings())), canon(t, f.Filter.Profile); got != want {
		t.Errorf("FilterProfileFields:\n got %s\nwant %s", got, want)
	}
}

func TestProfileRoundtrip(t *testing.T) {
	f := loadProfilesFixture(t)
	var p ModelProfile
	if err := json.Unmarshal(f.ProfileRoundtrip, &p); err != nil {
		t.Fatalf("unmarshal profile: %v", err)
	}
	p.Normalize()
	if got, want := marshalCanon(t, p), canon(t, f.ProfileRoundtrip); got != want {
		t.Errorf("profile roundtrip:\n got %s\nwant %s", got, want)
	}
}

func TestTemplateRoundtrip(t *testing.T) {
	f := loadProfilesFixture(t)
	var g GlobalTemplate
	if err := json.Unmarshal(f.TemplateRoundtrip, &g); err != nil {
		t.Fatalf("unmarshal template: %v", err)
	}
	g.Normalize()
	if got, want := marshalCanon(t, g), canon(t, f.TemplateRoundtrip); got != want {
		t.Errorf("template roundtrip:\n got %s\nwant %s", got, want)
	}
}

func TestMinimalFromDict(t *testing.T) {
	f := loadProfilesFixture(t)
	minimal := []byte(`{"name":"minimal","display_name":"Minimal","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00"}`)
	var p ModelProfile
	if err := json.Unmarshal(minimal, &p); err != nil {
		t.Fatalf("unmarshal minimal: %v", err)
	}
	p.Normalize()
	if got, want := marshalCanon(t, p), canon(t, f.MinimalRoundtrip); got != want {
		t.Errorf("minimal roundtrip:\n got %s\nwant %s", got, want)
	}
}

func TestValidateProfileName(t *testing.T) {
	f := loadProfilesFixture(t)
	for _, v := range f.Validations {
		err := ValidateProfileName(v.Name)
		if v.Valid && err != nil {
			t.Errorf("name %q: expected valid, got error %v", v.Name, err)
		}
		if !v.Valid && err == nil {
			t.Errorf("name %q: expected invalid, got no error", v.Name)
		}
	}
}

func BenchmarkFilterProfileFields(b *testing.B) {
	s := sampleSettings()
	b.ReportAllocs()
	for b.Loop() {
		_ = FilterProfileFields(s)
	}
}
