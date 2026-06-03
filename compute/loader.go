// SPDX-License-Identifier: MIT OR Apache-2.0

package compute

import (
	"fmt"

	"github.com/tamnd/fastmlx/mlxgo"
)

// mlxgoDtype maps a safetensors dtype tag to the mlxgo dtype the
// loaded array carries. The float8 tags have no mlx equivalent and are
// reported unsupported.
func mlxgoDtype(tag string) (mlxgo.Dtype, bool) {
	switch tag {
	case "BOOL":
		return mlxgo.Bool, true
	case "U8":
		return mlxgo.Uint8, true
	case "I8":
		return mlxgo.Int8, true
	case "U16":
		return mlxgo.Uint16, true
	case "I16":
		return mlxgo.Int16, true
	case "F16":
		return mlxgo.Float16, true
	case "BF16":
		return mlxgo.BFloat16, true
	case "U32":
		return mlxgo.Uint32, true
	case "I32":
		return mlxgo.Int32, true
	case "F32":
		return mlxgo.Float32, true
	case "U64":
		return mlxgo.Uint64, true
	case "I64":
		return mlxgo.Int64, true
	case "F64":
		return mlxgo.Float64, true
	default:
		return 0, false
	}
}

// TensorBytes returns the raw data slice of one tensor inside a safetensors
// blob: the bytes at [DataStart+Begin, DataStart+End). It bounds-checks the
// range against the blob so a truncated file is caught rather than panicking.
func TensorBytes(blob []byte, h *SafetensorsHeader, t TensorInfo) ([]byte, error) {
	start := h.DataStart + t.Begin
	end := h.DataStart + t.End
	if end < start || end > uint64(len(blob)) {
		return nil, fmt.Errorf("safetensors: tensor %q range [%d,%d) out of bounds for %d-byte blob",
			t.Name, start, end, len(blob))
	}
	return blob[start:end], nil
}

// LoadTensors parses a safetensors blob and builds the name-to-array weight map
// a model is assembled from. Each tensor's bytes are handed to mlxgo unchanged,
// so float16 and bfloat16 weights load without a host-side conversion. The
// __metadata__ entry is not a tensor and is skipped by the header parser.
func LoadTensors(blob []byte) (map[string]*mlxgo.Array, error) {
	h, err := ParseSafetensorsHeader(blob)
	if err != nil {
		return nil, err
	}
	weights := make(map[string]*mlxgo.Array, len(h.Tensors))
	for _, t := range h.Tensors {
		dtype, ok := mlxgoDtype(t.Dtype)
		if !ok {
			return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %q", t.Name, t.Dtype)
		}
		raw, err := TensorBytes(blob, h, t)
		if err != nil {
			return nil, err
		}
		arr, err := mlxgo.NewBytes(raw, dtype, t.Shape...)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", t.Name, err)
		}
		weights[t.Name] = arr
	}
	return weights, nil
}

// MergeTensors combines the weight maps of a sharded checkpoint into one. A
// name appearing in more than one shard is a malformed index and is reported.
func MergeTensors(shards ...map[string]*mlxgo.Array) (map[string]*mlxgo.Array, error) {
	total := 0
	for _, s := range shards {
		total += len(s)
	}
	out := make(map[string]*mlxgo.Array, total)
	for _, s := range shards {
		for name, arr := range s {
			if _, dup := out[name]; dup {
				return nil, fmt.Errorf("safetensors: weight %q present in more than one shard", name)
			}
			out[name] = arr
		}
	}
	return out, nil
}
