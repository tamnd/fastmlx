// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"testing"
)

type verifyCase struct {
	APIKey string `json:"api_key"`
	Server string `json:"server"`
	Out    bool   `json:"out"`
}

type verifyAnyCase struct {
	APIKey  string   `json:"api_key"`
	Main    string   `json:"main"`
	SubKeys []string `json:"sub_keys"`
	Out     bool     `json:"out"`
}

type validateCase struct {
	APIKey string `json:"api_key"`
	Ok     bool   `json:"ok"`
	Msg    string `json:"msg"`
}

type authFixture struct {
	Verify    []verifyCase    `json:"verify"`
	VerifyAny []verifyAnyCase `json:"verify_any"`
	Validate  []validateCase  `json:"validate"`
}

func loadAuth(t *testing.T) authFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/auth.json")
	if err != nil {
		t.Fatal(err)
	}
	var f authFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestVerifyAPIKeyParity(t *testing.T) {
	for i, c := range loadAuth(t).Verify {
		if got := VerifyAPIKey(c.APIKey, c.Server); got != c.Out {
			t.Errorf("VerifyAPIKey case %d (%q,%q) = %v, want %v", i, c.APIKey, c.Server, got, c.Out)
		}
	}
}

func TestVerifyAnyAPIKeyParity(t *testing.T) {
	for i, c := range loadAuth(t).VerifyAny {
		if got := VerifyAnyAPIKey(c.APIKey, c.Main, c.SubKeys); got != c.Out {
			t.Errorf("VerifyAnyAPIKey case %d (%q,%q,%v) = %v, want %v", i, c.APIKey, c.Main, c.SubKeys, got, c.Out)
		}
	}
}

func TestValidateAPIKeyParity(t *testing.T) {
	for i, c := range loadAuth(t).Validate {
		ok, msg := ValidateAPIKey(c.APIKey)
		if ok != c.Ok || msg != c.Msg {
			t.Errorf("ValidateAPIKey case %d (%q) = (%v,%q), want (%v,%q)", i, c.APIKey, ok, msg, c.Ok, c.Msg)
		}
	}
}

func BenchmarkValidateAPIKey(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ValidateAPIKey("valid-key-123")
	}
}
