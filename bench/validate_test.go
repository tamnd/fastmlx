// SPDX-License-Identifier: MIT OR Apache-2.0

package bench

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type detectValidateFixture struct {
	ValidPromptLengths []int `json:"valid_prompt_lengths"`
	ValidBatchSizes    []int `json:"valid_batch_sizes"`
	Detect             []struct {
		Config  *string `json:"config"`
		Dirname string  `json:"dirname"`
		Result  string  `json:"result"`
	} `json:"detect"`
	PromptValidation []struct {
		In  []int `json:"in"`
		Out struct {
			Error  *string `json:"error"`
			Sorted []int   `json:"sorted"`
		} `json:"out"`
	} `json:"prompt_validation"`
	BatchValidation []struct {
		In  []int `json:"in"`
		Out struct {
			Error  *string `json:"error"`
			Sorted []int   `json:"sorted"`
		} `json:"out"`
	} `json:"batch_validation"`
}

func loadDetectValidate(t *testing.T) detectValidateFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/detect_validate.json")
	if err != nil {
		t.Fatal(err)
	}
	var f detectValidateFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func numberMap(t *testing.T, s *string) map[string]any {
	t.Helper()
	if s == nil {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(*s)))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		t.Fatalf("decode %s: %v", *s, err)
	}
	return m
}

func TestValidSets(t *testing.T) {
	f := loadDetectValidate(t)
	if !reflect.DeepEqual(VALIDPromptLengths, f.ValidPromptLengths) {
		t.Errorf("VALIDPromptLengths = %v, want %v", VALIDPromptLengths, f.ValidPromptLengths)
	}
	if !reflect.DeepEqual(VALIDBatchSizes, f.ValidBatchSizes) {
		t.Errorf("VALIDBatchSizes = %v, want %v", VALIDBatchSizes, f.ValidBatchSizes)
	}
}

func TestDetectQuantizationParity(t *testing.T) {
	for _, c := range loadDetectValidate(t).Detect {
		config := numberMap(t, c.Config)
		if got := DetectQuantization(config, c.Dirname); got != c.Result {
			t.Errorf("DetectQuantization(%v, %q) = %q, want %q", c.Config, c.Dirname, got, c.Result)
		}
	}
}

func TestValidatePromptLengthsParity(t *testing.T) {
	for _, c := range loadDetectValidate(t).PromptValidation {
		got, err := ValidatePromptLengths(c.In)
		checkValidation(t, "prompt", c.In, got, err, c.Out.Error, c.Out.Sorted)
	}
}

func TestValidateBatchSizesParity(t *testing.T) {
	for _, c := range loadDetectValidate(t).BatchValidation {
		got, err := ValidateBatchSizes(c.In)
		checkValidation(t, "batch", c.In, got, err, c.Out.Error, c.Out.Sorted)
	}
}

func checkValidation(t *testing.T, kind string, in, got []int, err error, wantErr *string, wantSorted []int) {
	t.Helper()
	if wantErr != nil {
		if err == nil {
			t.Errorf("%s %v: expected error %q, got nil", kind, in, *wantErr)
		} else if err.Error() != *wantErr {
			t.Errorf("%s %v: error = %q, want %q", kind, in, err.Error(), *wantErr)
		}
		return
	}
	if err != nil {
		t.Errorf("%s %v: unexpected error %v", kind, in, err)
		return
	}
	if len(got) == 0 && len(wantSorted) == 0 {
		return
	}
	if !reflect.DeepEqual(got, wantSorted) {
		t.Errorf("%s %v: sorted = %v, want %v", kind, in, got, wantSorted)
	}
}

func BenchmarkDetectQuantization(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = DetectQuantization(nil, "Qwen3-30B-A3B-4bit")
	}
}

func BenchmarkValidatePromptLengths(b *testing.B) {
	in := []int{200000, 8192, 1024}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ValidatePromptLengths(in)
	}
}
