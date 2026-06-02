// SPDX-License-Identifier: MIT OR Apache-2.0

package recovery

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
)

type recoveryFixture struct {
	Patterns []string `json:"patterns"`
	Cases    []struct {
		Msg     string `json:"msg"`
		Corrupt bool   `json:"corrupt"`
	} `json:"cases"`
}

func loadFixture(t *testing.T) recoveryFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/recovery.json")
	if err != nil {
		t.Fatal(err)
	}
	var f recoveryFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCorruptionPatternsParity(t *testing.T) {
	fx := loadFixture(t)
	if !reflect.DeepEqual(CorruptionPatterns, fx.Patterns) {
		t.Errorf("pattern set diverged:\n got  %q\n want %q", CorruptionPatterns, fx.Patterns)
	}
}

func TestIsCorruptionMessageParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.Cases {
		if got := IsCorruptionMessage(c.Msg); got != c.Corrupt {
			t.Errorf("IsCorruptionMessage(%q) = %v, want %v", c.Msg, got, c.Corrupt)
		}
	}
}

func TestIsCorruptionErrorParity(t *testing.T) {
	fx := loadFixture(t)
	for _, c := range fx.Cases {
		if got := IsCorruptionError(errors.New(c.Msg)); got != c.Corrupt {
			t.Errorf("IsCorruptionError(%q) = %v, want %v", c.Msg, got, c.Corrupt)
		}
	}
}

func TestIsCorruptionErrorNil(t *testing.T) {
	if IsCorruptionError(nil) {
		t.Error("nil error must not be corruption")
	}
}

func BenchmarkIsCorruptionError(b *testing.B) {
	// A miss walks the whole pattern list, the worst case.
	err := errors.New("some unrelated error message that matches nothing at all")
	b.ReportAllocs()
	for b.Loop() {
		_ = IsCorruptionError(err)
	}
}
