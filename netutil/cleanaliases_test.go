// SPDX-License-Identifier: MIT OR Apache-2.0

package netutil

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type cleanAliasesFixture struct {
	Clean []struct {
		In  []string `json:"in"`
		Out struct {
			Cleaned []string `json:"cleaned"`
			Error   string   `json:"error"`
		} `json:"out"`
	} `json:"clean"`
	Repr []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	} `json:"repr"`
}

func loadCleanAliases(t *testing.T) cleanAliasesFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/cleanaliases.json")
	if err != nil {
		t.Fatal(err)
	}
	var f cleanAliasesFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCleanServerAliasesParity(t *testing.T) {
	for i, c := range loadCleanAliases(t).Clean {
		cleaned, errDetail := CleanServerAliases(c.In)
		if c.Out.Error != "" {
			if errDetail != c.Out.Error {
				t.Errorf("clean case %d (%v): errDetail\n got  %q\n want %q", i, c.In, errDetail, c.Out.Error)
			}
			if cleaned != nil {
				t.Errorf("clean case %d (%v): expected nil cleaned on error, got %v", i, c.In, cleaned)
			}
			continue
		}
		if errDetail != "" {
			t.Errorf("clean case %d (%v): unexpected errDetail %q", i, c.In, errDetail)
		}
		want := c.Out.Cleaned
		if want == nil {
			want = []string{}
		}
		if !reflect.DeepEqual(cleaned, want) {
			t.Errorf("clean case %d (%v):\n got  %v\n want %v", i, c.In, cleaned, want)
		}
	}
}

func TestPyStrReprParity(t *testing.T) {
	for i, c := range loadCleanAliases(t).Repr {
		if got := pyStrRepr(c.In); got != c.Out {
			t.Errorf("pyStrRepr case %d (%q):\n got  %s\n want %s", i, c.In, got, c.Out)
		}
	}
}

func BenchmarkCleanServerAliases(b *testing.B) {
	aliases := []string{"  host1  ", "host1", "host2.local", "192.168.1.10", "10.0.0.1"}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = CleanServerAliases(aliases)
	}
}
