// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/tamnd/fastmlx/mlxgo"
)

type loaderFixture struct {
	BlobB64   string `json:"blob_b64"`
	DataStart uint64 `json:"data_start"`
	HasBF16   bool   `json:"has_bf16"`
	Expected  map[string]struct {
		Dtype    string `json:"dtype"`
		Shape    []int  `json:"shape"`
		BytesB64 string `json:"bytes_b64"`
	} `json:"expected"`
}

func loadLoaderFixture(t *testing.T) (loaderFixture, []byte) {
	t.Helper()
	raw, err := os.ReadFile("testdata/loader_tensors.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f loaderFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	blob, err := base64.StdEncoding.DecodeString(f.BlobB64)
	if err != nil {
		t.Fatalf("decode blob: %v", err)
	}
	return f, blob
}

func TestTensorBytesParity(t *testing.T) {
	f, blob := loadLoaderFixture(t)
	h, err := ParseSafetensorsHeader(blob)
	if err != nil {
		t.Fatalf("ParseSafetensorsHeader: %v", err)
	}
	if h.DataStart != f.DataStart {
		t.Errorf("DataStart = %d, want %d", h.DataStart, f.DataStart)
	}
	for name, exp := range f.Expected {
		info, ok := h.Tensor(name)
		if !ok {
			t.Fatalf("tensor %q absent from header", name)
		}
		got, err := TensorBytes(blob, h, info)
		if err != nil {
			t.Fatalf("TensorBytes(%q): %v", name, err)
		}
		want, err := base64.StdEncoding.DecodeString(exp.BytesB64)
		if err != nil {
			t.Fatalf("decode want bytes: %v", err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("tensor %q bytes mismatch: got %v want %v", name, got, want)
		}
	}
}

func TestTensorBytesOutOfBounds(t *testing.T) {
	_, blob := loadLoaderFixture(t)
	h, err := ParseSafetensorsHeader(blob)
	if err != nil {
		t.Fatalf("ParseSafetensorsHeader: %v", err)
	}
	// A tensor whose range runs past a truncated blob must error, not panic.
	bad := TensorInfo{Name: "x", Dtype: "F32", Shape: []int{4}, Begin: 0, End: 16}
	if _, err := TensorBytes(blob[:h.DataStart+4], h, bad); err == nil {
		t.Error("expected an out-of-bounds error")
	}
}

func TestLoadTensorsParity(t *testing.T) {
	f, blob := loadLoaderFixture(t)
	if !f.HasBF16 {
		t.Fatal("fixture should include a bfloat16 tensor to exercise the no-Go-scalar path")
	}
	weights, err := LoadTensors(blob)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	if len(weights) != len(f.Expected) {
		t.Errorf("loaded %d tensors, want %d", len(weights), len(f.Expected))
	}
	dtypeTag := map[mlxgo.Dtype]string{
		mlxgo.Float32: "F32", mlxgo.Int32: "I32", mlxgo.BFloat16: "BF16",
	}
	for name, exp := range f.Expected {
		arr, ok := weights[name]
		if !ok {
			t.Errorf("weight %q not loaded", name)
			continue
		}
		if !reflect.DeepEqual(arr.Shape(), exp.Shape) {
			t.Errorf("%q shape = %v, want %v", name, arr.Shape(), exp.Shape)
		}
		if tag := dtypeTag[arr.Dtype()]; tag != exp.Dtype {
			t.Errorf("%q dtype = %v (%s), want %s", name, arr.Dtype(), tag, exp.Dtype)
		}
	}
}

func TestLoadTensorsUnsupportedDtype(t *testing.T) {
	if _, ok := mlxgoDtype("F8_E5M2"); ok {
		t.Error("float8 should be reported unsupported")
	}
	for tag, want := range map[string]mlxgo.Dtype{
		"BOOL": mlxgo.Bool, "F16": mlxgo.Float16, "BF16": mlxgo.BFloat16,
		"F32": mlxgo.Float32, "I64": mlxgo.Int64, "U8": mlxgo.Uint8,
	} {
		got, ok := mlxgoDtype(tag)
		if !ok || got != want {
			t.Errorf("mlxgoDtype(%q) = %v,%v, want %v,true", tag, got, ok, want)
		}
	}
}

func TestMergeTensors(t *testing.T) {
	a, _ := mlxgo.NewFloat32([]float32{1}, 1)
	b, _ := mlxgo.NewFloat32([]float32{2}, 1)
	merged, err := MergeTensors(
		map[string]*mlxgo.Array{"w1": a},
		map[string]*mlxgo.Array{"w2": b},
	)
	if err != nil {
		t.Fatalf("MergeTensors: %v", err)
	}
	if len(merged) != 2 {
		t.Errorf("merged %d, want 2", len(merged))
	}
	if _, err := MergeTensors(
		map[string]*mlxgo.Array{"dup": a},
		map[string]*mlxgo.Array{"dup": b},
	); err == nil {
		t.Error("expected a duplicate-weight error")
	}
}

func BenchmarkLoadTensors(b *testing.B) {
	raw, err := os.ReadFile("testdata/loader_tensors.json")
	if err != nil {
		b.Fatal(err)
	}
	var f loaderFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		b.Fatal(err)
	}
	blob, err := base64.StdEncoding.DecodeString(f.BlobB64)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := LoadTensors(blob); err != nil {
			b.Fatal(err)
		}
	}
}
