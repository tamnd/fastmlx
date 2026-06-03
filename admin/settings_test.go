// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type settingsFixture struct {
	SSEKeepalive []struct {
		Mode   string `json:"mode"`
		Detail string `json:"detail"`
	} `json:"sse_keepalive"`
	ModelDirs []struct {
		ModelDirs *[]string `json:"model_dirs"`
		ModelDir  *string   `json:"model_dir"`
		Result    struct {
			HasUpdate bool      `json:"has_update"`
			NewDirs   *[]string `json:"new_dirs"`
			Primary   *string   `json:"primary"`
		} `json:"result"`
	} `json:"model_dirs"`
}

func loadSettings(t *testing.T) settingsFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/settings.json")
	if err != nil {
		t.Fatal(err)
	}
	var f settingsFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestValidateSSEKeepaliveModeParity(t *testing.T) {
	for i, c := range loadSettings(t).SSEKeepalive {
		if got := ValidateSSEKeepaliveMode(c.Mode); got != c.Detail {
			t.Errorf("case %d (%q): got %q want %q", i, c.Mode, got, c.Detail)
		}
	}
}

func TestResolveModelDirsParity(t *testing.T) {
	for i, c := range loadSettings(t).ModelDirs {
		got := ResolveModelDirs(c.ModelDirs, c.ModelDir)
		if got.HasUpdate != c.Result.HasUpdate {
			t.Errorf("case %d has_update: got %v want %v", i, got.HasUpdate, c.Result.HasUpdate)
		}
		// new_dirs: null in the fixture means no update; otherwise a list.
		var wantDirs []string
		if c.Result.NewDirs != nil {
			wantDirs = *c.Result.NewDirs
		}
		if !sameDirs(got.NewDirs, wantDirs) {
			t.Errorf("case %d new_dirs:\n got  %#v\n want %#v", i, got.NewDirs, wantDirs)
		}
		if !samePtr(got.Primary, c.Result.Primary) {
			t.Errorf("case %d primary: got %v want %v", i, deref(got.Primary), deref(c.Result.Primary))
		}
	}
}

// sameDirs compares two directory lists treating nil and empty as equal, since
// the resolver returns a non-nil empty slice for an all-blank list.
func sameDirs(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func samePtr(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func deref(p *string) any {
	if p == nil {
		return nil
	}
	return *p
}

func BenchmarkResolveModelDirs(b *testing.B) {
	dirs := []string{"  /models/a  ", "", "/models/b", "   "}
	b.ReportAllocs()
	for b.Loop() {
		_ = ResolveModelDirs(&dirs, nil)
	}
}

func BenchmarkValidateSSEKeepaliveMode(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = ValidateSSEKeepaliveMode("bogus")
	}
}
