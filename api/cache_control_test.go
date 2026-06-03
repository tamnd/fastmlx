// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"testing"
)

type cacheControlFixture struct {
	Cases []struct {
		Request json.RawMessage `json:"request"`
		Has     bool            `json:"has"`
	} `json:"cases"`
}

func loadCacheControl(t *testing.T) cacheControlFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/cache_control.json")
	if err != nil {
		t.Fatal(err)
	}
	var f cacheControlFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestRequestHasCacheControlParity(t *testing.T) {
	for i, c := range loadCacheControl(t).Cases {
		req, ok := parseOrdered(string(c.Request))
		if !ok {
			t.Fatalf("case %d: bad request fixture", i)
		}
		if got := RequestHasCacheControl(req); got != c.Has {
			t.Errorf("case %d: got %v, want %v", i, got, c.Has)
		}
	}
}

func BenchmarkRequestHasCacheControl(b *testing.B) {
	req, _ := parseOrdered(`{"system":"s","tools":[{"name":"t"}],"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"one"}]},` +
		`{"role":"assistant","content":[{"type":"text","text":"two","cache_control":{"type":"ephemeral"}}]}]}`)
	b.ReportAllocs()
	for b.Loop() {
		_ = RequestHasCacheControl(req)
	}
}
