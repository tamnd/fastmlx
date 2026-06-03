// SPDX-License-Identifier: MIT OR Apache-2.0

package install

import (
	"encoding/json"
	"os"
	"testing"
)

type installFixture struct {
	IsAppBundle []struct {
		Here string `json:"here"`
		Out  bool   `json:"out"`
	} `json:"is_app_bundle"`
	AppBundleCLIPath []struct {
		Here string `json:"here"`
		Out  string `json:"out"`
	} `json:"app_bundle_cli_path"`
	UserCLIShimPath []struct {
		Home string `json:"home"`
		Out  string `json:"out"`
	} `json:"user_cli_shim_path"`
	IsHomebrew []struct {
		Prefix string `json:"prefix"`
		Out    bool   `json:"out"`
	} `json:"is_homebrew"`
	InstallMethod []struct {
		Here   string `json:"here"`
		Prefix string `json:"prefix"`
		Out    string `json:"out"`
	} `json:"install_method"`
	CLIPrefix []struct {
		Here               string `json:"here"`
		UserShimExecutable bool   `json:"user_shim_executable"`
		Out                string `json:"out"`
	} `json:"cli_prefix"`
}

func loadInstall(t *testing.T) installFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/install.json")
	if err != nil {
		t.Fatal(err)
	}
	var f installFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestInstallParity(t *testing.T) {
	f := loadInstall(t)
	for i, c := range f.IsAppBundle {
		if got := IsAppBundle(c.Here); got != c.Out {
			t.Errorf("IsAppBundle case %d (%q): got %v want %v", i, c.Here, got, c.Out)
		}
	}
	for i, c := range f.AppBundleCLIPath {
		if got := AppBundleCLIPath(c.Here); got != c.Out {
			t.Errorf("AppBundleCLIPath case %d:\n got  %q\n want %q", i, got, c.Out)
		}
	}
	for i, c := range f.UserCLIShimPath {
		if got := UserCLIShimPath(c.Home); got != c.Out {
			t.Errorf("UserCLIShimPath case %d: got %q want %q", i, got, c.Out)
		}
	}
	for i, c := range f.IsHomebrew {
		if got := IsHomebrew(c.Prefix); got != c.Out {
			t.Errorf("IsHomebrew case %d (%q): got %v want %v", i, c.Prefix, got, c.Out)
		}
	}
	for i, c := range f.InstallMethod {
		if got := InstallMethod(c.Here, c.Prefix); got != c.Out {
			t.Errorf("InstallMethod case %d: got %q want %q", i, got, c.Out)
		}
	}
	for i, c := range f.CLIPrefix {
		if got := CLIPrefix(c.Here, c.UserShimExecutable); got != c.Out {
			t.Errorf("CLIPrefix case %d:\n got  %q\n want %q", i, got, c.Out)
		}
	}
}

func BenchmarkCLIPrefix(b *testing.B) {
	here := "/Applications/fastmlx.app/Contents/Resources/lib/python3.13/site-packages/fastmlx/utils/install.py"
	b.ReportAllocs()
	for b.Loop() {
		_ = CLIPrefix(here, false)
	}
}

func BenchmarkInstallMethod(b *testing.B) {
	here := "/opt/homebrew/Cellar/fastmlx/0.1.0/libexec/lib/python3.13/site-packages/fastmlx/utils/install.py"
	prefix := "/opt/homebrew/Cellar/python/3.13"
	b.ReportAllocs()
	for b.Loop() {
		_ = InstallMethod(here, prefix)
	}
}
