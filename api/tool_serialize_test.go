// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"sort"
	"testing"
)

type toolSerializeFixture struct {
	Serialize []struct {
		Kind string          `json:"kind"`
		In   json.RawMessage `json:"in"`
		Out  string          `json:"out"`
	} `json:"serialize"`
	Names []struct {
		In  json.RawMessage `json:"in"`
		Out []string        `json:"out"`
	} `json:"names"`
}

func loadToolSerializeFixture(t *testing.T) toolSerializeFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/tool_serialize.json")
	if err != nil {
		t.Fatal(err)
	}
	var f toolSerializeFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSerializeToolCallArgumentsParity(t *testing.T) {
	for i, c := range loadToolSerializeFixture(t).Serialize {
		var arg jval
		if c.Kind == "str" {
			var s string
			if err := json.Unmarshal(c.In, &s); err != nil {
				t.Fatalf("case %d: bad string input: %v", i, err)
			}
			arg = jval{kind: kindString, s: s}
		} else {
			v, ok := parseOrdered(string(c.In))
			if !ok {
				t.Fatalf("case %d: bad input fixture %s", i, c.In)
			}
			arg = v
		}
		if got := SerializeToolCallArguments(arg); got != c.Out {
			t.Errorf("case %d (%s): SerializeToolCallArguments(%s) = %q, want %q", i, c.Kind, c.In, got, c.Out)
		}
	}
}

func TestExtractToolNamesParity(t *testing.T) {
	for i, c := range loadToolSerializeFixture(t).Names {
		arr, ok := parseOrdered(string(c.In))
		if !ok || arr.kind != kindArray {
			t.Fatalf("case %d: bad input fixture %s", i, c.In)
		}
		set := ExtractToolNames(arr.arr)
		got := make([]string, 0, len(set))
		for name := range set {
			got = append(got, name)
		}
		sort.Strings(got)
		if len(got) != len(c.Out) {
			t.Errorf("case %d: ExtractToolNames(%s) = %v, want %v", i, c.In, got, c.Out)
			continue
		}
		for j := range got {
			if got[j] != c.Out[j] {
				t.Errorf("case %d: ExtractToolNames(%s) = %v, want %v", i, c.In, got, c.Out)
				break
			}
		}
	}
}

func BenchmarkSerializeToolCallArguments(b *testing.B) {
	arg, _ := parseOrdered(`{"location":"Tokyo","unit":"celsius","count":3,"ratio":1.50}`)
	str := jval{kind: kindString, s: `{"location":"Tokyo","unit":"celsius","count":3,"ratio":1.50}`}
	b.ReportAllocs()
	for b.Loop() {
		_ = SerializeToolCallArguments(arg)
		_ = SerializeToolCallArguments(str)
	}
}

func BenchmarkExtractToolNames(b *testing.B) {
	arr, _ := parseOrdered(`[{"function":{"name":"a"}},{"function":{"name":"b"}},{"function":{"name":"c"}}]`)
	b.ReportAllocs()
	for b.Loop() {
		_ = ExtractToolNames(arr.arr)
	}
}
