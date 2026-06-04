// SPDX-License-Identifier: MIT OR Apache-2.0

package models

import (
	"errors"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

// The concrete model forwards must be assignable to SequenceForward, which is the
// whole premise of the adapter: a model value is a forward without a wrapper.
var (
	_ SequenceForward = (&Qwen3Model{}).Forward
	_ SequenceForward = (&LlamaModel{}).Forward
	_ SequenceForward = (&Glm4Model{}).Forward
	_ SequenceForward = (&Phi4Model{}).Forward
	_ SequenceForward = (&Ministral3Model{}).Forward
)

func TestAdapterNewCache(t *testing.T) {
	a := NewAdapter(4, 2, nil, nil)
	cache := a.NewCache()
	caches, ok := cache.([]*KVTensorCache)
	if !ok {
		t.Fatalf("NewCache returned %T, want []*KVTensorCache", cache)
	}
	if len(caches) != 4 {
		t.Fatalf("NewCache made %d caches, want 4", len(caches))
	}
	for i, c := range caches {
		if c == nil {
			t.Fatalf("cache[%d] is nil", i)
		}
		if c.Offset != 0 {
			t.Fatalf("cache[%d] offset = %d, want 0", i, c.Offset)
		}
		if c.Keys() != nil || c.Values() != nil {
			t.Fatalf("cache[%d] is not empty", i)
		}
	}
	// Two NewCache calls hand out independent slices.
	other := a.NewCache().([]*KVTensorCache)
	if &caches[0] == &other[0] {
		t.Fatal("NewCache shared its backing array between calls")
	}
}

func TestAdapterForwardPassesTokensAndCache(t *testing.T) {
	var gotTokens []int32
	var gotCaches []*KVTensorCache
	a := NewAdapter(3, 0, func(tokens []int32, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
		gotTokens = tokens
		gotCaches = caches
		// Return a well-shaped logits array so the seam (the last-row gather) is
		// what stops the host build, not a shape error.
		return mlxgo.NewFloat32([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	}, nil)
	cache := a.NewCache()
	_, err := a.Forward([]int32{7, 8}, cache, nil)
	if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("Forward err = %v, want ErrMLXUnavailable from the last-row seam", err)
	}
	if len(gotTokens) != 2 || gotTokens[0] != 7 || gotTokens[1] != 8 {
		t.Fatalf("forward saw tokens %v, want [7 8]", gotTokens)
	}
	if len(gotCaches) != 3 {
		t.Fatalf("forward saw %d caches, want 3", len(gotCaches))
	}
}

func TestAdapterForwardCacheTypeGuard(t *testing.T) {
	called := false
	a := NewAdapter(2, 0, func([]int32, []*KVTensorCache, *mlxgo.Stream) (*mlxgo.Array, error) {
		called = true
		return nil, nil
	}, nil)
	_, err := a.Forward([]int32{1}, 1234, nil) // wrong cache type
	if err == nil {
		t.Fatal("Forward accepted a non-cache value")
	}
	if called {
		t.Fatal("forward ran despite the cache-type guard failing")
	}
}

func TestAdapterForwardErrorPropagates(t *testing.T) {
	sentinel := errors.New("boom")
	a := NewAdapter(1, 0, func([]int32, []*KVTensorCache, *mlxgo.Stream) (*mlxgo.Array, error) {
		return nil, sentinel
	}, nil)
	_, err := a.Forward([]int32{1}, a.NewCache(), nil)
	if !errors.Is(err, sentinel) {
		t.Fatalf("Forward err = %v, want the model's error", err)
	}
}

func TestAdapterEOS(t *testing.T) {
	if got := NewAdapter(1, 42, nil, nil).EOS(); got != 42 {
		t.Fatalf("EOS = %d, want 42", got)
	}
}

func TestAdapterBatchDecodeReachesSeam(t *testing.T) {
	var gotTokens []int32
	var gotBatch int
	a := NewAdapter(2, 0, nil, func(tokens []int32, batch int, caches []*KVTensorCache, s *mlxgo.Stream) (*mlxgo.Array, error) {
		gotTokens = tokens
		gotBatch = batch
		// Well-shaped [batch, 1, vocab] logits so the seam is the per-row gather,
		// not a shape error: the empty caches merge to nils without a kernel, the
		// forward returns this host array, and batchRows hits the first Take.
		return mlxgo.NewFloat32([]float32{1, 2, 3, 4, 5, 6}, batch, 1, 3)
	})
	caches := []any{a.NewCache(), a.NewCache()}
	_, err := a.BatchDecode([]int32{5, 6}, caches, nil)
	if !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("BatchDecode err = %v, want ErrMLXUnavailable from the row-gather seam", err)
	}
	if gotBatch != 2 {
		t.Fatalf("batched forward saw batch %d, want 2", gotBatch)
	}
	if len(gotTokens) != 2 || gotTokens[0] != 5 || gotTokens[1] != 6 {
		t.Fatalf("batched forward saw tokens %v, want [5 6]", gotTokens)
	}
}

func TestAdapterBatchDecodeCacheTypeGuard(t *testing.T) {
	called := false
	a := NewAdapter(1, 0, nil, func([]int32, int, []*KVTensorCache, *mlxgo.Stream) (*mlxgo.Array, error) {
		called = true
		return nil, nil
	})
	_, err := a.BatchDecode([]int32{1, 2}, []any{a.NewCache(), 1234}, nil) // second cache wrong type
	if err == nil {
		t.Fatal("BatchDecode accepted a non-cache value")
	}
	if called {
		t.Fatal("batched forward ran despite the cache-type guard failing")
	}
}

func TestBatchRowsDimGuard(t *testing.T) {
	flat, err := mlxgo.NewFloat32([]float32{1, 2, 3, 4}, 2, 2) // 2-D, not [batch, 1, vocab]
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	if _, err := batchRows(flat, 2, nil); err == nil {
		t.Fatal("batchRows accepted a 2-D array")
	} else if errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("batchRows reached the kernel seam on a bad shape: %v", err)
	}
}

func TestLastRowDimGuard(t *testing.T) {
	flat, err := mlxgo.NewFloat32([]float32{1, 2, 3, 4}, 2, 2) // 2-D, not [1, L, vocab]
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	if _, err := lastRow(flat, nil); err == nil {
		t.Fatal("lastRow accepted a 2-D array")
	} else if errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("lastRow reached the kernel seam on a bad shape: %v", err)
	}
}

func TestLastRowSeam(t *testing.T) {
	logits, err := mlxgo.NewFloat32([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	if err != nil {
		t.Fatalf("NewFloat32: %v", err)
	}
	if _, err := lastRow(logits, nil); !errors.Is(err, mlxgo.ErrMLXUnavailable) {
		t.Fatalf("lastRow err = %v, want ErrMLXUnavailable", err)
	}
}

func BenchmarkAdapterNewCache(b *testing.B) {
	a := NewAdapter(48, 0, nil, nil)
	b.ReportAllocs()
	for b.Loop() {
		_ = a.NewCache()
	}
}
