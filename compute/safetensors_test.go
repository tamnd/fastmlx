// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type stTruth struct {
	HeaderSize uint64            `json:"header_size"`
	DataStart  uint64            `json:"data_start"`
	DataLen    uint64            `json:"data_len"`
	Metadata   map[string]string `json:"metadata"`
	Tensors    []struct {
		Name  string `json:"name"`
		Dtype string `json:"dtype"`
		Shape []int  `json:"shape"`
		Begin uint64 `json:"begin"`
		End   uint64 `json:"end"`
	} `json:"tensors"`
}

func TestParseSafetensorsHeaderParity(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "safetensors_header.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Label   string  `json:"label"`
		BlobB64 string  `json:"blob_b64"`
		Truth   stTruth `json:"truth"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	for _, c := range cases {
		blob, err := base64.StdEncoding.DecodeString(c.BlobB64)
		if err != nil {
			t.Fatalf("%s: decode blob: %v", c.Label, err)
		}
		h, err := ParseSafetensorsHeader(blob)
		if err != nil {
			t.Fatalf("%s: parse: %v", c.Label, err)
		}
		if h.HeaderSize != c.Truth.HeaderSize {
			t.Errorf("%s: header_size = %d, want %d", c.Label, h.HeaderSize, c.Truth.HeaderSize)
		}
		if h.DataStart != c.Truth.DataStart {
			t.Errorf("%s: data_start = %d, want %d", c.Label, h.DataStart, c.Truth.DataStart)
		}
		if len(h.Tensors) != len(c.Truth.Tensors) {
			t.Fatalf("%s: tensor count = %d, want %d", c.Label, len(h.Tensors), len(c.Truth.Tensors))
		}
		for i, want := range c.Truth.Tensors {
			got := h.Tensors[i]
			if got.Name != want.Name || got.Dtype != want.Dtype || got.Begin != want.Begin || got.End != want.End {
				t.Errorf("%s: tensor %d = {%s %s [%d,%d)}, want {%s %s [%d,%d)}",
					c.Label, i, got.Name, got.Dtype, got.Begin, got.End,
					want.Name, want.Dtype, want.Begin, want.End)
			}
			wantShape := want.Shape
			if wantShape == nil {
				wantShape = []int{}
			}
			gotShape := got.Shape
			if gotShape == nil {
				gotShape = []int{}
			}
			if !reflect.DeepEqual(gotShape, wantShape) {
				t.Errorf("%s: tensor %d shape = %v, want %v", c.Label, i, gotShape, wantShape)
			}
			// ByName must round-trip to the same record.
			if bn, ok := h.Tensor(want.Name); !ok || bn.Begin != want.Begin {
				t.Errorf("%s: Tensor(%q) lookup failed", c.Label, want.Name)
			}
		}
		if c.Truth.Metadata == nil {
			if h.Metadata != nil {
				t.Errorf("%s: metadata = %v, want nil", c.Label, h.Metadata)
			}
		} else if !reflect.DeepEqual(h.Metadata, c.Truth.Metadata) {
			t.Errorf("%s: metadata = %v, want %v", c.Label, h.Metadata, c.Truth.Metadata)
		}
	}
}

// buildBlob assembles a minimal safetensors blob from a header JSON string and a
// data buffer length, for the error and round-trip cases.
func buildBlob(headerJSON string, dataLen int) []byte {
	out := make([]byte, 8+len(headerJSON)+dataLen)
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerJSON)))
	copy(out[8:], headerJSON)
	return out
}

func TestParseSafetensorsHeaderErrors(t *testing.T) {
	if _, err := ParseSafetensorsHeader([]byte{0, 1, 2}); err != ErrShortBuffer {
		t.Errorf("short buffer: err = %v, want ErrShortBuffer", err)
	}
	// declared header longer than the buffer
	big := make([]byte, 8)
	binary.LittleEndian.PutUint64(big, 1000)
	if _, err := ParseSafetensorsHeader(big); err != ErrHeaderTooLarge {
		t.Errorf("header too large: err = %v, want ErrHeaderTooLarge", err)
	}
	// unknown dtype
	blob := buildBlob(`{"w":{"dtype":"Q4","shape":[2],"data_offsets":[0,8]}}`, 8)
	if _, err := ParseSafetensorsHeader(blob); err == nil {
		t.Error("unknown dtype: expected error")
	}
	// byte length disagrees with shape*dtype (2 f32 = 8 bytes, claims 16)
	blob = buildBlob(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}`, 16)
	if _, err := ParseSafetensorsHeader(blob); err == nil {
		t.Error("size mismatch: expected error")
	}
	// range exceeds data buffer
	blob = buildBlob(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 4)
	if _, err := ParseSafetensorsHeader(blob); err == nil {
		t.Error("range overflow: expected error")
	}
	// begin > end
	blob = buildBlob(`{"w":{"dtype":"F32","shape":[0],"data_offsets":[8,0]}}`, 8)
	if _, err := ParseSafetensorsHeader(blob); err == nil {
		t.Error("begin>end: expected error")
	}
}

func TestParseSafetensorsIndex(t *testing.T) {
	doc := `{"metadata":{"total_size":1024},"weight_map":{
		"model.layers.0.w":"model-00001-of-00002.safetensors",
		"model.layers.1.w":"model-00002-of-00002.safetensors",
		"model.embed":"model-00001-of-00002.safetensors"}}`
	idx, err := ParseSafetensorsIndex([]byte(doc))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	wantShards := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	if !reflect.DeepEqual(idx.Shards, wantShards) {
		t.Errorf("shards = %v, want %v", idx.Shards, wantShards)
	}
	if f, ok := idx.ShardFor("model.embed"); !ok || f != "model-00001-of-00002.safetensors" {
		t.Errorf("ShardFor(embed) = %q,%v", f, ok)
	}
	if _, ok := idx.ShardFor("missing"); ok {
		t.Error("ShardFor(missing) should be false")
	}
	if _, err := ParseSafetensorsIndex([]byte(`{"weight_map":{}}`)); err == nil {
		t.Error("empty weight_map: expected error")
	}
}

func BenchmarkParseSafetensorsHeader(b *testing.B) {
	b.ReportAllocs()
	blob := buildBlob(`{"embed.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},`+
		`"lm_head.bias":{"dtype":"I32","shape":[4],"data_offsets":[64,80]}}`, 80)
	for b.Loop() {
		if _, err := ParseSafetensorsHeader(blob); err != nil {
			b.Fatal(err)
		}
	}
}
