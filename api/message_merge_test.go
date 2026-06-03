// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type mergeRolesFixture struct {
	Merge []struct {
		In  json.RawMessage `json:"in"`
		Out json.RawMessage `json:"out"`
	} `json:"merge"`
}

func loadMergeRoles(t *testing.T) mergeRolesFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/merge_roles.json")
	if err != nil {
		t.Fatal(err)
	}
	var f mergeRolesFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestMergeConsecutiveRolesParity(t *testing.T) {
	for i, c := range loadMergeRoles(t).Merge {
		in, ok := parseOrdered(string(c.In))
		if !ok {
			t.Fatalf("case %d: bad input fixture", i)
		}
		want, ok := parseOrdered(string(c.Out))
		if !ok {
			t.Fatalf("case %d: bad output fixture", i)
		}
		got := jval{kind: kindArray, arr: mergeConsecutiveRoles(in.arr)}
		if got.dumpASCII() != want.dumpASCII() {
			t.Errorf("case %d:\n got  %s\n want %s", i, got.dumpASCII(), want.dumpASCII())
		}
	}
}

func TestMergeConsecutiveRolesEmpty(t *testing.T) {
	if got := mergeConsecutiveRoles(nil); got != nil {
		t.Errorf("nil input: got %v, want nil", got)
	}
	if got := mergeConsecutiveRoles([]jval{}); len(got) != 0 {
		t.Errorf("empty input: got %v, want empty", got)
	}
}

func TestMergeConsecutiveRolesNoMutateInput(t *testing.T) {
	in, _ := parseOrdered(`[{"role":"user","content":"a"},{"role":"user","content":"b"}]`)
	before := in.dumpASCII()
	_ = mergeConsecutiveRoles(in.arr)
	if in.dumpASCII() != before {
		t.Errorf("input mutated:\n got  %s\n want %s", in.dumpASCII(), before)
	}
}

func BenchmarkMergeConsecutiveRoles(b *testing.B) {
	in, _ := parseOrdered(`[{"role":"user","content":"one"},{"role":"user","content":"two"},` +
		`{"role":"assistant","content":"three"},{"role":"user","content":"four"}]`)
	msgs := in.arr
	b.ReportAllocs()
	for b.Loop() {
		_ = mergeConsecutiveRoles(msgs)
	}
}
