// SPDX-License-Identifier: MIT OR Apache-2.0

package registry

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// fakeOwner stands in for an engine. alive models the weak-reference probe
// (flipped off to simulate garbage collection); resets counts forced transfers.
type fakeOwner struct {
	alive  bool
	resets int
}

func (f *fakeOwner) Alive() bool { return f.alive }
func (f *fakeOwner) Reset()      { f.resets++ }

type step struct {
	Op string `json:"op"`

	Model  string `json:"model"`
	Engine string `json:"engine"`
	Force  bool   `json:"force"`
	OK     bool   `json:"ok"`
	Raised bool   `json:"raised"`

	Owned bool    `json:"owned"`
	Owner *string `json:"owner"`

	TotalEntries int `json:"total_entries"`
	ActiveOwners int `json:"active_owners"`
	Cleaned      int `json:"cleaned"`
	Resets       int `json:"resets"`
}

func loadSteps(t *testing.T) []step {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "registry.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		Steps []step `json:"steps"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx.Steps
}

func TestRegistrySequence(t *testing.T) {
	steps := loadSteps(t)
	reg := New()
	// One owner per engine id, created live; "gc" flips a flag off.
	owners := map[string]*fakeOwner{}
	owner := func(id string) *fakeOwner {
		o, ok := owners[id]
		if !ok {
			o = &fakeOwner{alive: true}
			owners[id] = o
		}
		return o
	}

	for i, s := range steps {
		switch s.Op {
		case "acquire":
			ok, err := reg.Acquire(s.Model, owner(s.Engine), s.Engine, s.Force)
			if ok != s.OK {
				t.Fatalf("step %d acquire ok = %v, want %v", i, ok, s.OK)
			}
			if raised := err != nil; raised != s.Raised {
				t.Fatalf("step %d acquire raised = %v, want %v", i, raised, s.Raised)
			}
			if s.Raised {
				if _, isOwnErr := err.(*ModelOwnershipError); !isOwnErr {
					t.Fatalf("step %d want *ModelOwnershipError, got %T", i, err)
				}
			}
		case "release":
			if ok := reg.Release(s.Model, s.Engine); ok != s.OK {
				t.Fatalf("step %d release ok = %v, want %v", i, ok, s.OK)
			}
		case "is_owned":
			owned, id := reg.IsOwned(s.Model)
			if owned != s.Owned {
				t.Fatalf("step %d is_owned owned = %v, want %v", i, owned, s.Owned)
			}
			wantOwner := ""
			if s.Owner != nil {
				wantOwner = *s.Owner
			}
			if id != wantOwner {
				t.Fatalf("step %d is_owned owner = %q, want %q", i, id, wantOwner)
			}
		case "gc":
			owner(s.Engine).alive = false
		case "stats":
			st := reg.Stats()
			if st.TotalEntries != s.TotalEntries || st.ActiveOwners != s.ActiveOwners {
				t.Fatalf("step %d stats = %+v, want total=%d active=%d", i, st, s.TotalEntries, s.ActiveOwners)
			}
		case "cleanup":
			if c := reg.Cleanup(); c != s.Cleaned {
				t.Fatalf("step %d cleanup = %d, want %d", i, c, s.Cleaned)
			}
		case "resets":
			if got := owner(s.Engine).resets; got != s.Resets {
				t.Fatalf("step %d resets[%s] = %d, want %d", i, s.Engine, got, s.Resets)
			}
		default:
			t.Fatalf("step %d unknown op %q", i, s.Op)
		}
	}
}

func TestModelOwnershipErrorMessage(t *testing.T) {
	err := &ModelOwnershipError{OwnerID: "engine-7"}
	want := "Model is already owned by engine engine-7. Use force=True or call release() on the previous engine first."
	if err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
}

func TestStatsJSONOrder(t *testing.T) {
	out, err := json.Marshal(Stats{TotalEntries: 3, ActiveOwners: 2})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got := string(out); got != `{"total_entries":3,"active_owners":2}` {
		t.Fatalf("stats json = %s", got)
	}
}

func TestDefaultSingleton(t *testing.T) {
	a, b := Default(), Default()
	if a != b {
		t.Fatal("Default returned distinct instances")
	}
}

func BenchmarkAcquireRelease(b *testing.B) {
	reg := New()
	o := &fakeOwner{alive: true}
	b.ReportAllocs()
	for b.Loop() {
		reg.Acquire("m", o, "A", false)
		reg.Release("m", "A")
	}
}
