// SPDX-License-Identifier: MIT OR Apache-2.0

package integrations

import (
	"encoding/json"
	"os"
	"testing"
)

type configfileFixture struct {
	Cases        []rawCase        `json:"cases"`
	Existing     []map[string]any `json:"existing"`
	OpenClaw     []map[string]any `json:"openclaw"`
	OpenCode     []map[string]any `json:"opencode"`
	PiModels     []map[string]any `json:"pi_models"`
	PiSettings   []map[string]any `json:"pi_settings"`
	OpenClawExec []struct {
		Profile string         `json:"profile"`
		Out     map[string]any `json:"out"`
	} `json:"openclaw_exec"`
}

func loadConfigfile(t *testing.T) configfileFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/configfile.json")
	if err != nil {
		t.Fatal(err)
	}
	var f configfileFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// cloneMap round-trips through JSON so a fresh, independent copy of the existing
// config is handed to each updater, the way the reference deep-copies before
// mutating.
func cloneMap(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// canonJSON marshals a value so two structurally equal configs compare equal
// regardless of key order or int-vs-float number typing.
func canonJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var any2 any
	if err := json.Unmarshal(b, &any2); err != nil {
		t.Fatal(err)
	}
	out, err := json.Marshal(any2)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func checkConfig(t *testing.T, name string, i int, got, want map[string]any) {
	t.Helper()
	if canonJSON(t, got) != canonJSON(t, want) {
		t.Errorf("%s case %d =\n  %s\nwant\n  %s", name, i, canonJSON(t, got), canonJSON(t, want))
	}
}

func TestOpenClawConfigParity(t *testing.T) {
	f := loadConfigfile(t)
	for i, rc := range f.Cases {
		got := OpenClawConfig(cloneMap(t, f.Existing[i]), rc.context())
		checkConfig(t, "OpenClawConfig", i, got, f.OpenClaw[i])
	}
}

func TestOpenCodeConfigParity(t *testing.T) {
	f := loadConfigfile(t)
	for i, rc := range f.Cases {
		got := OpenCodeConfig(cloneMap(t, f.Existing[i]), rc.context())
		checkConfig(t, "OpenCodeConfig", i, got, f.OpenCode[i])
	}
}

func TestPiModelsParity(t *testing.T) {
	f := loadConfigfile(t)
	for i, rc := range f.Cases {
		got := PiModels(cloneMap(t, f.Existing[i]), rc.context())
		checkConfig(t, "PiModels", i, got, f.PiModels[i])
	}
}

func TestPiSettingsParity(t *testing.T) {
	f := loadConfigfile(t)
	for i, rc := range f.Cases {
		got := PiSettings(cloneMap(t, f.Existing[i]), rc.context())
		checkConfig(t, "PiSettings", i, got, f.PiSettings[i])
	}
}

func TestOpenClawExecApprovalsParity(t *testing.T) {
	f := loadConfigfile(t)
	for i, c := range f.OpenClawExec {
		got := OpenClawExecApprovals(map[string]any{}, c.Profile)
		checkConfig(t, "OpenClawExecApprovals", i, got, c.Out)
	}
}

func BenchmarkOpenClawConfig(b *testing.B) {
	cw, mt := 32768, 4096
	c := Context{Host: "127.0.0.1", Port: 8000, APIKey: "k", Model: "Qwen3-8B", ContextWindow: &cw, MaxTokens: &mt, ToolsProfile: "coding"}
	b.ReportAllocs()
	for b.Loop() {
		_ = OpenClawConfig(map[string]any{}, c)
	}
}

func BenchmarkOpenCodeConfig(b *testing.B) {
	cw := 32768
	c := Context{Host: "127.0.0.1", Port: 8000, Model: "Qwen3-8B", ContextWindow: &cw}
	b.ReportAllocs()
	for b.Loop() {
		_ = OpenCodeConfig(map[string]any{}, c)
	}
}
