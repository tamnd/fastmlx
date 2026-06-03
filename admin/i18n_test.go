// SPDX-License-Identifier: MIT OR Apache-2.0

package admin

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type staticCase struct {
	Path   string  `json:"path"`
	Mtime  float64 `json:"mtime"`
	IsFile bool    `json:"is_file"`
	Out    string  `json:"out"`
}

type translateCase struct {
	Key string `json:"key"`
	Out any    `json:"out"`
}

type loadLocaleCase struct {
	Primary  string         `json:"primary"`
	Fallback string         `json:"fallback"`
	Out      map[string]any `json:"out"`
}

type i18nFixture struct {
	Static    []staticCase     `json:"static"`
	Translate []translateCase  `json:"translate"`
	Load      []loadLocaleCase `json:"load"`
}

func loadI18n(t *testing.T) i18nFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/i18n.json")
	if err != nil {
		t.Fatal(err)
	}
	var f i18nFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestStaticVersionURLParity(t *testing.T) {
	for i, c := range loadI18n(t).Static {
		if got := StaticVersionURL(c.Path, c.Mtime, c.IsFile); got != c.Out {
			t.Errorf("StaticVersionURL case %d (%q, %v) = %q, want %q", i, c.Path, c.Mtime, got, c.Out)
		}
	}
}

func TestTranslateKeyParity(t *testing.T) {
	locale := map[string]any{"hello": "Xin chao", "bye": "Tam biet", "count": "So luong"}
	for i, c := range loadI18n(t).Translate {
		got := TranslateKey(locale, c.Key)
		if !reflect.DeepEqual(got, c.Out) {
			t.Errorf("TranslateKey case %d (%q) = %v, want %v", i, c.Key, got, c.Out)
		}
	}
}

func TestLoadLocaleTextParity(t *testing.T) {
	for i, c := range loadI18n(t).Load {
		got := LoadLocaleText(c.Primary, c.Fallback)
		want := c.Out
		if want == nil {
			want = map[string]any{}
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("LoadLocaleText case %d:\n got  %v\n want %v", i, got, want)
		}
	}
}

func BenchmarkStaticVersionURL(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = StaticVersionURL("css/admin.css", 1700000000.9, true)
	}
}

func BenchmarkLoadLocaleText(b *testing.B) {
	primary := `{"hello": "Xin chao", "bye": "Tam biet", "count": "So luong"}`
	b.ReportAllocs()
	for b.Loop() {
		_ = LoadLocaleText(primary, "{}")
	}
}
