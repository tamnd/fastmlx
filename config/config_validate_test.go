// SPDX-License-Identifier: MIT OR Apache-2.0

package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type validateCase struct {
	Label  string   `json:"label"`
	Errors []string `json:"errors"`
}

type validateFixture struct {
	Validate []validateCase `json:"validate"`
}

// mutate applies the same change the reference capture made for a given label,
// starting from a default Config.
func mutate(label string, c *Config) {
	switch label {
	case "defaults_valid":
	case "port_zero":
		c.Server.Port = 0
	case "port_too_high":
		c.Server.Port = 70000
	case "port_negative":
		c.Server.Port = -5
	case "max_tokens_zero":
		c.Generation.MaxTokens = 0
	case "temp_low":
		c.Generation.Temperature = -0.1
	case "temp_high":
		c.Generation.Temperature = 2.5
	case "top_p_high":
		c.Generation.TopP = 1.5
	case "cache_enabled_no_dir":
		c.PagedSSDCache.Enabled = true
	case "cache_enabled_with_dir":
		c.PagedSSDCache.Enabled = true
		c.PagedSSDCache.CacheDir = "/tmp/x"
	case "many_errors":
		c.Server.Port = 99999
		c.Generation.MaxTokens = -3
		c.Generation.Temperature = 5.0
		c.Generation.TopP = -0.2
		c.PagedSSDCache.Enabled = true
	case "temp_zero_ok":
		c.Generation.Temperature = 0.0
	case "temp_two_ok":
		c.Generation.Temperature = 2.0
	case "top_p_one_ok":
		c.Generation.TopP = 1.0
	case "top_p_zero_ok":
		c.Generation.TopP = 0.0
	case "port_one_ok":
		c.Server.Port = 1
	case "port_max_ok":
		c.Server.Port = 65535
	default:
		panic("unknown case label: " + label)
	}
}

func loadValidateFixture(t *testing.T) validateFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "config_validate.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx validateFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	if len(fx.Validate) == 0 {
		t.Fatal("fixture has no cases")
	}
	return fx
}

func TestValidateParity(t *testing.T) {
	fx := loadValidateFixture(t)
	for _, tc := range fx.Validate {
		t.Run(tc.Label, func(t *testing.T) {
			c := DefaultConfig()
			mutate(tc.Label, &c)
			got := c.Validate()
			if len(got) == 0 && len(tc.Errors) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.Errors) {
				t.Fatalf("errors mismatch\n got: %#v\nwant: %#v", got, tc.Errors)
			}
		})
	}
}

func TestValidateAlwaysNonNil(t *testing.T) {
	if got := DefaultConfig().Validate(); got == nil {
		t.Fatal("Validate returned nil; want empty non-nil slice")
	}
}

func TestPyFloat(t *testing.T) {
	cases := map[float64]string{
		5.0:   "5.0",
		2.5:   "2.5",
		-0.1:  "-0.1",
		-0.2:  "-0.2",
		1.5:   "1.5",
		0.0:   "0.0",
		2.0:   "2.0",
		1.0:   "1.0",
		0.95:  "0.95",
		100.0: "100.0",
	}
	for in, want := range cases {
		if got := pyFloat(in); got != want {
			t.Errorf("pyFloat(%v) = %q, want %q", in, got, want)
		}
	}
}

func BenchmarkValidate(b *testing.B) {
	c := DefaultConfig()
	c.Server.Port = 99999
	c.Generation.MaxTokens = -3
	c.Generation.Temperature = 5.0
	c.Generation.TopP = -0.2
	c.PagedSSDCache.Enabled = true
	b.ReportAllocs()
	for b.Loop() {
		_ = c.Validate()
	}
}
