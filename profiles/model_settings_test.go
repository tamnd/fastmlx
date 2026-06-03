// SPDX-License-Identifier: MIT OR Apache-2.0

package profiles

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type msToDictCase struct {
	Label  string          `json:"label"`
	Input  json.RawMessage `json:"input"`
	ToDict json.RawMessage `json:"to_dict"`
}

type msValidateCase struct {
	Label string  `json:"label"`
	Error *string `json:"error"`
}

type msFixture struct {
	Versions map[string]int   `json:"versions"`
	ToDict   []msToDictCase   `json:"to_dict"`
	FromDict []msToDictCase   `json:"from_dict"`
	Validate []msValidateCase `json:"validate"`
}

func loadMSFixture(t *testing.T) msFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "model_settings.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx msFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

// applyInput unmarshals an input object over a fresh defaults struct, the same
// path FromModelSettingsDict takes (minus validation).
func applyInput(t *testing.T, input json.RawMessage) ModelSettings {
	t.Helper()
	m := NewModelSettings()
	if len(input) > 0 {
		if err := json.Unmarshal(input, &m); err != nil {
			t.Fatalf("apply input: %v", err)
		}
	}
	return m
}

func TestModelSettingsVersions(t *testing.T) {
	fx := loadMSFixture(t)
	want := map[string]int{
		"SETTINGS_VERSION":  SettingsVersion,
		"PROFILES_VERSION":  ProfilesVersion,
		"TEMPLATES_VERSION": TemplatesVersion,
	}
	for k, v := range want {
		if fx.Versions[k] != v {
			t.Errorf("%s = %d, want %d", k, fx.Versions[k], v)
		}
	}
}

func TestModelSettingsToDict(t *testing.T) {
	fx := loadMSFixture(t)
	for _, tc := range fx.ToDict {
		t.Run(tc.Label, func(t *testing.T) {
			m := applyInput(t, tc.Input)
			got, err := m.ToDict()
			if err != nil {
				t.Fatalf("ToDict: %v", err)
			}
			if c := canon(t, got); c != canon(t, tc.ToDict) {
				t.Fatalf("to_dict mismatch\n got: %s\nwant: %s", c, canon(t, tc.ToDict))
			}
		})
	}
}

func TestModelSettingsFromDict(t *testing.T) {
	fx := loadMSFixture(t)
	for _, tc := range fx.FromDict {
		t.Run(tc.Label, func(t *testing.T) {
			m, err := FromModelSettingsDict(tc.Input)
			if err != nil {
				t.Fatalf("FromModelSettingsDict: %v", err)
			}
			got, err := m.ToDict()
			if err != nil {
				t.Fatalf("ToDict: %v", err)
			}
			if c := canon(t, got); c != canon(t, tc.ToDict) {
				t.Fatalf("roundtrip mismatch\n got: %s\nwant: %s", c, canon(t, tc.ToDict))
			}
		})
	}
}

func TestModelSettingsValidate(t *testing.T) {
	fx := loadMSFixture(t)
	for _, tc := range fx.Validate {
		t.Run(tc.Label, func(t *testing.T) {
			// Each validate case carries no explicit input map; rebuild the
			// flag combination from the label.
			m := NewModelSettings()
			switch tc.Label {
			case "ok_default":
			case "ok_mtp_alone":
				m.MTPEnabled = true
			case "ok_vlm_mtp_alone":
				m.VLMMTPEnabled = true
			case "mtp_and_dflash":
				m.MTPEnabled, m.DflashEnabled = true, true
			case "mtp_and_turboquant":
				m.MTPEnabled, m.TurboquantKVEnabled = true, true
			case "vlm_mtp_and_dflash":
				m.VLMMTPEnabled, m.DflashEnabled = true, true
			case "vlm_mtp_and_specprefill":
				m.VLMMTPEnabled, m.SpecprefillEnabled = true, true
			case "vlm_mtp_and_mtp":
				m.VLMMTPEnabled, m.MTPEnabled = true, true
			case "vlm_mtp_and_turboquant":
				m.VLMMTPEnabled, m.TurboquantKVEnabled = true, true
			default:
				t.Fatalf("unknown validate label: %s", tc.Label)
			}
			err := m.Validate()
			switch {
			case tc.Error == nil && err != nil:
				t.Fatalf("Validate() = %q, want nil", err)
			case tc.Error == nil && err == nil:
				// ok
			case tc.Error != nil && err == nil:
				t.Fatalf("Validate() = nil, want %q", *tc.Error)
			case err.Error() != *tc.Error:
				t.Fatalf("Validate() = %q, want %q", err.Error(), *tc.Error)
			}
		})
	}
}

func TestFromModelSettingsDictRejectsConflict(t *testing.T) {
	_, err := FromModelSettingsDict([]byte(`{"mtp_enabled":true,"dflash_enabled":true}`))
	if err == nil {
		t.Fatal("want conflict error, got nil")
	}
}

func BenchmarkModelSettingsToDict(b *testing.B) {
	m := NewModelSettings()
	temp := 0.7
	m.Temperature = &temp
	m.MTPEnabled = true
	b.ReportAllocs()
	for b.Loop() {
		_, _ = m.ToDict()
	}
}
