// SPDX-License-Identifier: MIT OR Apache-2.0

package auth

import (
	"encoding/json"
	"os"
	"testing"
)

type authFixture struct {
	Constants struct {
		SessionMaxAge    int `json:"session_max_age"`
		RememberMeMaxAge int `json:"remember_me_max_age"`
	} `json:"constants"`
	Validate []struct {
		Key   string `json:"key"`
		Valid bool   `json:"valid"`
		Error string `json:"error"`
	} `json:"validate"`
	VerifyAPIKey []struct {
		Key    string `json:"key"`
		Server string `json:"server"`
		OK     bool   `json:"ok"`
	} `json:"verify_api_key"`
	VerifyAny []struct {
		Key  string   `json:"key"`
		Main string   `json:"main"`
		Subs []string `json:"subs"`
		OK   bool     `json:"ok"`
	} `json:"verify_any"`
}

func loadFixture(t *testing.T) authFixture {
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

func TestConstantsParity(t *testing.T) {
	fx := loadFixture(t)
	if SessionMaxAge != fx.Constants.SessionMaxAge {
		t.Errorf("SessionMaxAge = %d, want %d", SessionMaxAge, fx.Constants.SessionMaxAge)
	}
	if RememberMeMaxAge != fx.Constants.RememberMeMaxAge {
		t.Errorf("RememberMeMaxAge = %d, want %d", RememberMeMaxAge, fx.Constants.RememberMeMaxAge)
	}
}

func TestValidateAPIKeyParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.Validate {
		valid, msg := ValidateAPIKey(c.Key)
		if valid != c.Valid || msg != c.Error {
			t.Errorf("ValidateAPIKey(%q) = (%v,%q), want (%v,%q)", c.Key, valid, msg, c.Valid, c.Error)
		}
	}
}

func TestVerifyAPIKeyParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.VerifyAPIKey {
		if got := VerifyAPIKey(c.Key, c.Server); got != c.OK {
			t.Errorf("VerifyAPIKey(%q,%q) = %v, want %v", c.Key, c.Server, got, c.OK)
		}
	}
}

func TestVerifyAnyAPIKeyParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.VerifyAny {
		subs := make([]SubKey, len(c.Subs))
		for i, s := range c.Subs {
			subs[i] = SubKey{Key: s}
		}
		if got := VerifyAnyAPIKey(c.Key, c.Main, subs); got != c.OK {
			t.Errorf("VerifyAnyAPIKey(%q,%q,%v) = %v, want %v", c.Key, c.Main, c.Subs, got, c.OK)
		}
	}
}

func TestSubKeyRoundTrip(t *testing.T) {
	// from_dict defaults missing fields to "", to_dict emits all three keys.
	var sk SubKey
	if err := json.Unmarshal([]byte(`{"key":"k1"}`), &sk); err != nil {
		t.Fatal(err)
	}
	if sk.Key != "k1" || sk.Name != "" || sk.CreatedAt != "" {
		t.Errorf("unmarshal partial = %+v", sk)
	}
	out, err := json.Marshal(SubKey{Key: "k1"})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"key":"k1","name":"","created_at":""}` {
		t.Errorf("marshal = %s", out)
	}
}

func BenchmarkValidateAPIKey(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ValidateAPIKey("a-reasonable-looking-api-key-1234")
	}
}
