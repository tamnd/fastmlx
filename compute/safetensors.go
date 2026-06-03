// SPDX-License-Identifier: MIT OR Apache-2.0

// Package compute holds the fastmlx inference backend: the safetensors model
// loader, KV caches, batch generator, sampler, and logits processors. The
// tensor kernels run on MLX through the mlxgo binding (built with cgo against
// mlx-c); the pieces in this file are the pure, host-testable parts of the
// loader that need no GPU — parsing the safetensors container header and the
// sharded-model index. The mmap and the per-weight mlx_array view are the one
// MLX seam and live in the cgo path.
package compute

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// dtypeSize maps a safetensors dtype tag to its element size in bytes. The set
// is exactly the one the safetensors format defines.
var dtypeSize = map[string]uint64{
	"BOOL":    1,
	"U8":      1,
	"I8":      1,
	"F8_E5M2": 1,
	"F8_E4M3": 1,
	"I16":     2,
	"U16":     2,
	"F16":     2,
	"BF16":    2,
	"I32":     4,
	"U32":     4,
	"F32":     4,
	"F64":     8,
	"I64":     8,
	"U64":     8,
}

// TensorInfo describes one weight in a safetensors container: its dtype tag,
// shape, and the [Begin, End) byte range of its data inside the data buffer
// (the bytes after the header, i.e. relative to DataStart).
type TensorInfo struct {
	Name  string
	Dtype string
	Shape []int
	Begin uint64
	End   uint64
}

// NumElements returns the product of the shape (1 for a scalar, 0 when any axis
// is zero).
func (t TensorInfo) NumElements() uint64 {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= uint64(d)
	}
	return n
}

// NumBytes returns the byte length of the tensor's data range.
func (t TensorInfo) NumBytes() uint64 { return t.End - t.Begin }

// SafetensorsHeader is the parsed container header. Tensors are listed in
// canonical order (by Begin offset, then name); ByName indexes into that slice.
// Metadata is the optional __metadata__ string map. DataStart is the byte offset
// where the data buffer begins (8 + HeaderSize).
type SafetensorsHeader struct {
	Tensors    []TensorInfo
	ByName     map[string]int
	Metadata   map[string]string
	HeaderSize uint64
	DataStart  uint64
}

// Tensor returns the named tensor's info and whether it was present.
func (h *SafetensorsHeader) Tensor(name string) (TensorInfo, bool) {
	i, ok := h.ByName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return h.Tensors[i], true
}

var (
	// ErrShortBuffer means the data is too small to hold the 8-byte length
	// prefix or the declared header.
	ErrShortBuffer = errors.New("safetensors: buffer shorter than declared header")
	// ErrHeaderTooLarge means the declared header length does not fit in the
	// buffer.
	ErrHeaderTooLarge = errors.New("safetensors: header length exceeds buffer")
)

type rawTensor struct {
	Dtype       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
}

// ParseSafetensorsHeader parses the safetensors container header out of the
// front of data: an 8-byte little-endian header length followed by that many
// bytes of JSON ({name: {dtype, shape, data_offsets}} plus an optional
// __metadata__ map). It validates that every tensor's dtype is known, that its
// byte range matches the shape and dtype, and that the ranges stay within the
// data buffer. The tensor data itself is not read.
func ParseSafetensorsHeader(data []byte) (*SafetensorsHeader, error) {
	if len(data) < 8 {
		return nil, ErrShortBuffer
	}
	headerSize := binary.LittleEndian.Uint64(data[:8])
	if headerSize > uint64(len(data)-8) {
		return nil, ErrHeaderTooLarge
	}
	headerJSON := data[8 : 8+headerSize]
	dataLen := uint64(len(data)) - 8 - headerSize

	// Decode preserving raw values so __metadata__ (a string map) can be
	// separated from the tensor entries (objects).
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: header JSON: %w", err)
	}

	h := &SafetensorsHeader{
		ByName:     make(map[string]int),
		HeaderSize: headerSize,
		DataStart:  8 + headerSize,
	}

	for name, rm := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(rm, &h.Metadata); err != nil {
				return nil, fmt.Errorf("safetensors: __metadata__: %w", err)
			}
			continue
		}
		var rt rawTensor
		if err := json.Unmarshal(rm, &rt); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		size, ok := dtypeSize[rt.Dtype]
		if !ok {
			return nil, fmt.Errorf("safetensors: tensor %q has unknown dtype %q", name, rt.Dtype)
		}
		begin, end := rt.DataOffsets[0], rt.DataOffsets[1]
		if begin > end {
			return nil, fmt.Errorf("safetensors: tensor %q has begin %d > end %d", name, begin, end)
		}
		if end > dataLen {
			return nil, fmt.Errorf("safetensors: tensor %q range [%d,%d) exceeds data length %d", name, begin, end, dataLen)
		}
		info := TensorInfo{Name: name, Dtype: rt.Dtype, Shape: rt.Shape, Begin: begin, End: end}
		if want := info.NumElements() * size; want != info.NumBytes() {
			return nil, fmt.Errorf("safetensors: tensor %q byte length %d does not match shape*dtype %d", name, info.NumBytes(), want)
		}
		h.Tensors = append(h.Tensors, info)
	}

	sort.Slice(h.Tensors, func(i, j int) bool {
		if h.Tensors[i].Begin != h.Tensors[j].Begin {
			return h.Tensors[i].Begin < h.Tensors[j].Begin
		}
		return h.Tensors[i].Name < h.Tensors[j].Name
	})
	for i, t := range h.Tensors {
		h.ByName[t.Name] = i
	}
	return h, nil
}
