// SPDX-License-Identifier: MIT OR Apache-2.0

package netutil

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

type aliasFixture struct {
	Hostname []struct {
		In  string `json:"in"`
		Out bool   `json:"out"`
	} `json:"hostname"`
	IP []struct {
		In  string `json:"in"`
		Out bool   `json:"out"`
	} `json:"ip"`
	Alias []struct {
		In  string `json:"in"`
		Out bool   `json:"out"`
	} `json:"alias"`
	Dedupe []struct {
		In  []string `json:"in"`
		Out []string `json:"out"`
	} `json:"dedupe"`
	Detect []struct {
		In struct {
			Host     string   `json:"host"`
			Hostname string   `json:"hostname"`
			FQDN     string   `json:"fqdn"`
			LocalIPs []string `json:"local_ips"`
		} `json:"in"`
		Out []string `json:"out"`
	} `json:"detect"`
}

func loadAliasFixture(t *testing.T) aliasFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/netalias.json")
	if err != nil {
		t.Fatal(err)
	}
	var f aliasFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestIsValidHostnameParity(t *testing.T) {
	fx := loadAliasFixture(t)
	for _, c := range fx.Hostname {
		if got := IsValidHostname(c.In); got != c.Out {
			t.Errorf("IsValidHostname(%q) = %v, want %v", c.In, got, c.Out)
		}
	}
}

func TestIsValidIPParity(t *testing.T) {
	fx := loadAliasFixture(t)
	for _, c := range fx.IP {
		if got := IsValidIP(c.In); got != c.Out {
			t.Errorf("IsValidIP(%q) = %v, want %v", c.In, got, c.Out)
		}
	}
}

func TestIsValidAliasParity(t *testing.T) {
	fx := loadAliasFixture(t)
	for _, c := range fx.Alias {
		if got := IsValidAlias(c.In); got != c.Out {
			t.Errorf("IsValidAlias(%q) = %v, want %v", c.In, got, c.Out)
		}
	}
}

func TestDedupePreserveOrderParity(t *testing.T) {
	fx := loadAliasFixture(t)
	for _, c := range fx.Dedupe {
		if got := dedupePreserveOrder(c.In); !slices.Equal(got, c.Out) {
			t.Errorf("dedupePreserveOrder(%v) = %v, want %v", c.In, got, c.Out)
		}
	}
}

func TestDetectServerAliasesParity(t *testing.T) {
	fx := loadAliasFixture(t)
	for _, c := range fx.Detect {
		got := DetectServerAliases(c.In.Host, SystemAliases{
			Hostname:  c.In.Hostname,
			FQDN:      c.In.FQDN,
			LocalIPv4: c.In.LocalIPs,
		})
		if !slices.Equal(got, c.Out) {
			t.Errorf("DetectServerAliases(%q, %+v) = %v, want %v", c.In.Host, c.In, got, c.Out)
		}
	}
}

func BenchmarkIsValidAlias(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsValidAlias("sub-domain.example.com")
		_ = IsValidAlias("192.168.1.5")
	}
}
