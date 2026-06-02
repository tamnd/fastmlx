// SPDX-License-Identifier: MIT OR Apache-2.0

package cache

import (
	"encoding/json"
	"os"
	"testing"
)

type blockhashFixture struct {
	Single []struct {
		Tokens   []int  `json:"tokens"`
		Model    string `json:"model"`
		Expected string `json:"expected"`
	} `json:"single"`
	Chains []struct {
		Tokens    []int    `json:"tokens"`
		BlockSize int      `json:"block_size"`
		Model     string   `json:"model"`
		Expected  []string `json:"expected"`
	} `json:"chains"`
}

func loadBlockhashFixture(t *testing.T) blockhashFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/blockhash.json")
	if err != nil {
		t.Fatal(err)
	}
	var f blockhashFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestComputeBlockHashDeterministic(t *testing.T) {
	fx := loadBlockhashFixture(t)
	for i, c := range fx.Single {
		got := ComputeBlockHash(nil, c.Tokens, c.Model).String()
		if got != c.Expected {
			t.Errorf("single[%d] tokens=%v model=%q:\n got  %s\n want %s", i, c.Tokens, c.Model, got, c.Expected)
		}
	}
}

func TestHashFullBlocksChained(t *testing.T) {
	fx := loadBlockhashFixture(t)
	for i, c := range fx.Chains {
		hashes := HashFullBlocks(c.Tokens, c.BlockSize, c.Model)
		if len(hashes) != len(c.Expected) {
			t.Fatalf("chain[%d]: got %d blocks, want %d", i, len(hashes), len(c.Expected))
		}
		for j, h := range hashes {
			if h.String() != c.Expected[j] {
				t.Errorf("chain[%d] block[%d]:\n got  %s\n want %s", i, j, h.String(), c.Expected[j])
			}
		}
	}
}

func TestComputeBlockHashChainSensitivity(t *testing.T) {
	// A block's hash must change with its parent: the same tokens after a
	// different prefix hash differently.
	p1 := ComputeBlockHash(nil, []int{1, 2}, "")
	p2 := ComputeBlockHash(nil, []int{9, 9}, "")
	a := ComputeBlockHash(&p1, []int{3, 4}, "")
	b := ComputeBlockHash(&p2, []int{3, 4}, "")
	if a == b {
		t.Error("same tokens under different parents should hash differently")
	}
	// Model name must isolate caches.
	m1 := ComputeBlockHash(nil, []int{1, 2}, "alpha")
	m2 := ComputeBlockHash(nil, []int{1, 2}, "beta")
	if m1 == m2 {
		t.Error("different model names should hash differently")
	}
}

func TestTupleRepr(t *testing.T) {
	cases := map[string][]int{
		"()":          {},
		"(5,)":        {5},
		"(1, 2, 3)":   {1, 2, 3},
		"(-1, 0, 42)": {-1, 0, 42},
	}
	for want, in := range cases {
		if got := tupleRepr(in); got != want {
			t.Errorf("tupleRepr(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestFindSharedPrefix(t *testing.T) {
	tokens := make([]int, 0, 260)
	for i := range 260 {
		tokens = append(tokens, i)
	}
	// 260 tokens at block 64 = 4 full blocks (256), 4 trailing.
	all := HashFullBlocks(tokens, 64, "m")
	if len(all) != 4 {
		t.Fatalf("expected 4 full blocks, got %d", len(all))
	}

	// Cache holds only the first two blocks: the prefix match stops at block 2.
	cached := map[BlockHash]bool{all[0]: true, all[1]: true}
	shared, numCached, remaining := FindSharedPrefix(tokens, 64, "m", func(h BlockHash) bool { return cached[h] })
	if len(shared) != 2 || numCached != 128 {
		t.Fatalf("shared=%d numCached=%d, want 2 / 128", len(shared), numCached)
	}
	if len(remaining) != 260-128 || remaining[0] != 128 {
		t.Fatalf("remaining len=%d first=%d, want %d / 128", len(remaining), remaining[0], 260-128)
	}

	// A gap in the middle is not bridged: a miss at block 0 yields no match even
	// if later blocks are present.
	cachedGap := map[BlockHash]bool{all[1]: true, all[2]: true, all[3]: true}
	shared2, numCached2, _ := FindSharedPrefix(tokens, 64, "m", func(h BlockHash) bool { return cachedGap[h] })
	if len(shared2) != 0 || numCached2 != 0 {
		t.Errorf("a missing first block should yield no prefix, got shared=%d numCached=%d", len(shared2), numCached2)
	}

	// Empty cache: nothing shared, everything remains.
	s3, n3, r3 := FindSharedPrefix(tokens, 64, "m", func(BlockHash) bool { return false })
	if len(s3) != 0 || n3 != 0 || len(r3) != 260 {
		t.Errorf("empty cache: shared=%d numCached=%d remaining=%d", len(s3), n3, len(r3))
	}
}

func TestHashFullBlocksGuards(t *testing.T) {
	if got := HashFullBlocks([]int{1, 2, 3}, 0, ""); got != nil {
		t.Error("non-positive block size should yield no hashes")
	}
	if got := HashFullBlocks([]int{1, 2, 3}, 64, ""); len(got) != 0 {
		t.Error("fewer tokens than a block should yield no full blocks")
	}
}

func BenchmarkFindSharedPrefix(b *testing.B) {
	tokens := make([]int, 4096)
	for i := range tokens {
		tokens[i] = i
	}
	cached := map[BlockHash]bool{}
	for _, h := range HashFullBlocks(tokens, 64, "m") {
		cached[h] = true
	}
	has := func(h BlockHash) bool { return cached[h] }
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = FindSharedPrefix(tokens, 64, "m", has)
	}
}
