// SPDX-License-Identifier: MIT OR Apache-2.0

package audioutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSplitTTSTextParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "split_tts_text.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text     string   `json:"text"`
		MaxChars int      `json:"max_chars"`
		Result   []string `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for i, c := range cases {
		got := SplitTTSText(c.Text, c.MaxChars)
		if len(got) != len(c.Result) {
			t.Errorf("case %d: SplitTTSText(%q, %d) = %#v want %#v",
				i, c.Text, c.MaxChars, got, c.Result)
			continue
		}
		for j := range got {
			if got[j] != c.Result[j] {
				t.Errorf("case %d chunk %d: got %q want %q (full got=%#v want=%#v)",
					i, j, got[j], c.Result[j], got, c.Result)
				break
			}
		}
	}
}

func BenchmarkSplitTTSText(b *testing.B) {
	b.ReportAllocs()
	text := "This clause is here, and this one too, and a third clause, plus a fourth, and finally a fifth segment to overflow."
	for b.Loop() {
		_ = SplitTTSText(text, 30)
	}
}
