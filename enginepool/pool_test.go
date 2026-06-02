// SPDX-License-Identifier: MIT OR Apache-2.0

package enginepool

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type poolFixture struct {
	FormatSize []struct {
		Bytes    int64  `json:"bytes"`
		Expected string `json:"expected"`
	} `json:"format_size"`
	Errors struct {
		TooLarge struct {
			ModelID   string `json:"model_id"`
			ModelSize int64  `json:"model_size"`
			Ceiling   int64  `json:"ceiling"`
			Expected  string `json:"expected"`
		} `json:"too_large"`
		NotFound struct {
			ModelID   string   `json:"model_id"`
			Available []string `json:"available"`
			Expected  string   `json:"expected"`
		} `json:"not_found"`
		NotFoundEmpty struct {
			ModelID   string   `json:"model_id"`
			Available []string `json:"available"`
			Expected  string   `json:"expected"`
		} `json:"not_found_empty"`
		Insufficient struct {
			ModelID   string `json:"model_id"`
			Projected int64  `json:"projected"`
			Ceiling   int64  `json:"ceiling"`
			Current   int64  `json:"current"`
			Model     int64  `json:"model"`
			Expected  string `json:"expected"`
		} `json:"insufficient"`
	} `json:"errors"`
}

func loadPoolFixture(t *testing.T) poolFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/pool.json")
	if err != nil {
		t.Fatal(err)
	}
	var f poolFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFormatSizeParity(t *testing.T) {
	fx := loadPoolFixture(t)
	for _, c := range fx.FormatSize {
		if got := FormatSize(c.Bytes); got != c.Expected {
			t.Errorf("FormatSize(%d) = %q, want %q", c.Bytes, got, c.Expected)
		}
	}
}

func TestErrorMessageParity(t *testing.T) {
	fx := loadPoolFixture(t)

	tl := &ModelTooLargeError{ModelID: fx.Errors.TooLarge.ModelID, ModelSize: fx.Errors.TooLarge.ModelSize, Ceiling: fx.Errors.TooLarge.Ceiling}
	if got := tl.Error(); got != fx.Errors.TooLarge.Expected {
		t.Errorf("ModelTooLargeError:\n got  %q\n want %q", got, fx.Errors.TooLarge.Expected)
	}

	nf := &ModelNotFoundError{ModelID: fx.Errors.NotFound.ModelID, Available: fx.Errors.NotFound.Available}
	if got := nf.Error(); got != fx.Errors.NotFound.Expected {
		t.Errorf("ModelNotFoundError:\n got  %q\n want %q", got, fx.Errors.NotFound.Expected)
	}

	nfe := &ModelNotFoundError{ModelID: fx.Errors.NotFoundEmpty.ModelID, Available: fx.Errors.NotFoundEmpty.Available}
	if got := nfe.Error(); got != fx.Errors.NotFoundEmpty.Expected {
		t.Errorf("ModelNotFoundError (empty):\n got  %q\n want %q", got, fx.Errors.NotFoundEmpty.Expected)
	}
}

// fakeEngine is a test stand-in whose active-request flag is controllable.
type fakeEngine struct{ active bool }

func (f *fakeEngine) HasActiveRequests() bool { return f.active }

// clock returns a controllable monotonic clock.
func clock() (func() float64, *float64) {
	t := 0.0
	now := func() float64 { t += 1.0; return t }
	return now, &t
}

func TestFindLRUVictimOldestNonPinned(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("a", "/a", 1, false)
	p.Register("b", "/b", 1, false)
	p.Register("c", "/c", 1, true) // pinned
	// Load a (t=1), then b (t=2), then c (t=3). a is oldest.
	p.MarkLoaded("a", &fakeEngine{})
	p.MarkLoaded("b", &fakeEngine{})
	p.MarkLoaded("c", &fakeEngine{})

	got, ok := p.FindLRUVictim()
	if !ok || got != "a" {
		t.Fatalf("victim = %q ok=%v, want a", got, ok)
	}

	// Touch a so b becomes the oldest.
	p.Touch("a")
	if got, _ := p.FindLRUVictim(); got != "b" {
		t.Errorf("after touch, victim = %q, want b", got)
	}
}

func TestFindLRUVictimSkipsActiveAndPinnedAndUnloaded(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("busy", "/busy", 1, false)
	p.Register("pinned", "/pinned", 1, true)
	p.Register("cold", "/cold", 1, false) // registered but never loaded
	p.MarkLoaded("busy", &fakeEngine{active: true})
	p.MarkLoaded("pinned", &fakeEngine{})

	if _, ok := p.FindLRUVictim(); ok {
		t.Error("no model should be evictable: busy has requests, pinned is pinned, cold is unloaded")
	}
}

func TestAdmitFitsWithoutEviction(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("m", "/m", 1000, false)
	err := p.Admit("m", 1_000_000, func() int64 { return 0 }, func(string) { t.Fatal("should not evict") })
	if err != nil {
		t.Fatalf("Admit = %v, want nil", err)
	}
}

func TestAdmitCeilingOffAdmitsUnconditionally(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("m", "/m", 1<<40, false)
	if err := p.Admit("m", 0, func() int64 { return 1 << 50 }, func(string) {}); err != nil {
		t.Fatalf("ceiling 0 should admit, got %v", err)
	}
}

func TestAdmitEvictsLRUUntilFits(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("old", "/old", 100, false)
	p.Register("new", "/new", 100, false)
	p.Register("want", "/want", 100, false)
	p.MarkLoaded("old", &fakeEngine{})
	p.MarkLoaded("new", &fakeEngine{})

	// Each loaded model "uses" 100; current memory is 100 * loaded count.
	loaded := map[string]bool{"old": true, "new": true}
	current := func() int64 {
		var n int64
		for m := range loaded {
			if loaded[m] {
				n += 100
			}
		}
		return n
	}
	var evicted []string
	unload := func(id string) { loaded[id] = false; evicted = append(evicted, id) }

	// Ceiling 250: current 200 + want 100 = 300 > 250, evict the LRU (old) ->
	// 100 + 100 = 200 <= 250, fits. Only old is evicted.
	err := p.Admit("want", 250, current, unload)
	if err != nil {
		t.Fatalf("Admit = %v, want nil", err)
	}
	if len(evicted) != 1 || evicted[0] != "old" {
		t.Fatalf("evicted = %v, want [old]", evicted)
	}
	if p.IsLoaded("old") {
		t.Error("old should be marked unloaded after eviction")
	}
}

func TestAdmitModelTooLarge(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("huge", "/huge", 1000, false)
	err := p.Admit("huge", 500, func() int64 { return 0 }, func(string) {})
	if _, ok := err.(*ModelTooLargeError); !ok {
		t.Fatalf("err = %T %v, want *ModelTooLargeError", err, err)
	}
}

func TestAdmitInsufficientMemoryParity(t *testing.T) {
	fx := loadPoolFixture(t)
	c := fx.Errors.Insufficient
	now, _ := clock()
	p := New(now)
	p.Register(c.ModelID, "/x", c.Model, false)
	// Nothing evictable; current leaves no room but the model alone fits.
	err := p.Admit(c.ModelID, c.Ceiling, func() int64 { return c.Current }, func(string) {})
	ime, ok := err.(*InsufficientMemoryError)
	if !ok {
		t.Fatalf("err = %T %v, want *InsufficientMemoryError", err, err)
	}
	if ime.Error() != c.Expected {
		t.Errorf("message:\n got  %q\n want %q", ime.Error(), c.Expected)
	}
}

func TestAdmitUnknownModel(t *testing.T) {
	now, _ := clock()
	p := New(now)
	p.Register("a", "/a", 1, false)
	err := p.Admit("ghost", 100, func() int64 { return 0 }, func(string) {})
	nf, ok := err.(*ModelNotFoundError)
	if !ok {
		t.Fatalf("err = %T, want *ModelNotFoundError", err)
	}
	if len(nf.Available) != 1 || nf.Available[0] != "a" {
		t.Errorf("Available = %v, want [a]", nf.Available)
	}
}

func TestEstimateModelSize(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "model-00001.safetensors"), 1000)
	writeFile(t, filepath.Join(dir, "model-00002.safetensors"), 2000)
	writeFile(t, filepath.Join(dir, "tokenizer.json"), 50) // not weights, ignored

	got, err := EstimateModelSize(dir)
	if err != nil {
		t.Fatal(err)
	}
	// (1000 + 2000) * 1.05 = 3150.
	if got != 3150 {
		t.Errorf("EstimateModelSize = %d, want 3150", got)
	}
}

func TestEstimateModelSizeBinFallbackSkipsOptimizer(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "pytorch_model.bin"), 4000)
	writeFile(t, filepath.Join(dir, "optimizer.bin"), 9999) // skipped

	got, err := EstimateModelSize(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != int64(float64(4000)*1.05) {
		t.Errorf("EstimateModelSize = %d, want %d", got, int64(float64(4000)*1.05))
	}
}

func TestEstimateModelSizeNoWeights(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), 10)
	if _, err := EstimateModelSize(dir); err == nil {
		t.Error("expected an error when no weight files are present")
	}
}

func writeFile(t *testing.T, path string, n int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, n), 0o644); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkFindLRUVictim(b *testing.B) {
	now, _ := clock()
	p := New(now)
	for i := range 16 {
		id := string(rune('a' + i))
		p.Register(id, "/"+id, 1, false)
		p.MarkLoaded(id, &fakeEngine{})
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = p.FindLRUVictim()
	}
}
