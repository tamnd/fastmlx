// SPDX-License-Identifier: MIT OR Apache-2.0

package release

import (
	"encoding/json"
	"os"
	"testing"
)

type releaseFixture struct {
	Parse []struct {
		In           string `json:"in"`
		Valid        bool   `json:"valid"`
		IsPrerelease bool   `json:"is_prerelease"`
		Str          string `json:"str"`
	} `json:"parse"`
	Compare []struct {
		A   string `json:"a"`
		B   string `json:"b"`
		Cmp int    `json:"cmp"`
	} `json:"compare"`
	Select []struct {
		Name     string    `json:"name"`
		Releases []Release `json:"releases"`
		OutTag   *string   `json:"out_tag"`
	} `json:"select"`
}

func loadReleaseFixture(t *testing.T) releaseFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/release.json")
	if err != nil {
		t.Fatal(err)
	}
	var f releaseFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestParseVersionParity(t *testing.T) {
	fx := loadReleaseFixture(t)
	for _, c := range fx.Parse {
		v, ok := ParseVersion(c.In)
		if ok != c.Valid {
			t.Errorf("ParseVersion(%q) valid = %v, want %v", c.In, ok, c.Valid)
			continue
		}
		if !ok {
			continue
		}
		if got := v.IsPrerelease(); got != c.IsPrerelease {
			t.Errorf("ParseVersion(%q).IsPrerelease() = %v, want %v", c.In, got, c.IsPrerelease)
		}
		if got := v.String(); got != c.Str {
			t.Errorf("ParseVersion(%q).String() = %q, want %q", c.In, got, c.Str)
		}
	}
}

func TestVersionCompareParity(t *testing.T) {
	fx := loadReleaseFixture(t)
	for _, c := range fx.Compare {
		a, ok := ParseVersion(c.A)
		if !ok {
			t.Fatalf("ParseVersion(%q) failed", c.A)
		}
		b, ok := ParseVersion(c.B)
		if !ok {
			t.Fatalf("ParseVersion(%q) failed", c.B)
		}
		if got := a.Compare(b); got != c.Cmp {
			t.Errorf("Compare(%q, %q) = %d, want %d", c.A, c.B, got, c.Cmp)
		}
		// Antisymmetry: reversing the operands negates the result.
		if got := b.Compare(a); got != -c.Cmp {
			t.Errorf("Compare(%q, %q) = %d, want %d", c.B, c.A, got, -c.Cmp)
		}
	}
}

func TestSelectLatestStableReleaseParity(t *testing.T) {
	fx := loadReleaseFixture(t)
	for _, c := range fx.Select {
		got := SelectLatestStableRelease(c.Releases)
		var gotTag *string
		if got != nil {
			gotTag = &got.TagName
		}
		switch {
		case gotTag == nil && c.OutTag == nil:
			// both none
		case gotTag == nil || c.OutTag == nil:
			t.Errorf("%s: SelectLatestStableRelease = %v, want %v", c.Name, derefOrNil(gotTag), derefOrNil(c.OutTag))
		case *gotTag != *c.OutTag:
			t.Errorf("%s: SelectLatestStableRelease = %q, want %q", c.Name, *gotTag, *c.OutTag)
		}
	}
}

func derefOrNil(s *string) any {
	if s == nil {
		return nil
	}
	return *s
}

func BenchmarkSelectLatestStableRelease(b *testing.B) {
	releases := []Release{
		{TagName: "v1.2.0"}, {TagName: "v1.10.0"}, {TagName: "v1.9.0rc1"},
		{TagName: "v1.9.0", Prerelease: true}, {TagName: "nightly"}, {TagName: "v2.0.0.dev3"},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = SelectLatestStableRelease(releases)
	}
}
