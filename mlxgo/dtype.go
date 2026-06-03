// SPDX-License-Identifier: MIT OR Apache-2.0

// Package mlxgo is the Go binding over the MLX C API that the compute backend
// runs its tensor kernels on. It builds in two modes selected by a build tag:
//
//   - Default (no tag): a pure-Go stub. Arrays carry shape, dtype, and host
//     data, so construction, metadata, and host round-trips work, but every op
//     that needs the GPU returns ErrMLXUnavailable. This is what CI and any host
//     without the MLX toolchain compile, so the serving layer and the pure
//     compute cores build and test everywhere.
//   - `-tags mlx`: the real cgo backend linking libmlxc / libmlx and the Metal,
//     Foundation, and Accelerate frameworks. Only this mode performs inference,
//     and it only compiles where the MLX dylibs and headers are installed
//     (see the `mlx` Makefile target).
//
// The two builds export the identical API; this file holds the parts that are
// pure data and shared by both — the dtype enum and the unavailable-backend
// sentinel.
package mlxgo

import (
	"errors"
	"fmt"
)

// ErrMLXUnavailable is returned by every compute operation in the default
// (stub) build. A build with `-tags mlx` and the MLX toolchain present performs
// the operation instead.
var ErrMLXUnavailable = errors.New("mlxgo: built without the mlx backend (rebuild with -tags mlx)")

// Dtype enumerates the array element types, ordered to match mlx-c's mlx_dtype
// enum so the cgo build maps each value to its C constant by position.
type Dtype int

const (
	Bool Dtype = iota
	Uint8
	Uint16
	Uint32
	Uint64
	Int8
	Int16
	Int32
	Int64
	Float16
	Float32
	BFloat16
	Complex64
	Float64
)

// String returns the dtype's short name (the safetensors-style tag).
func (d Dtype) String() string {
	switch d {
	case Bool:
		return "bool"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Float16:
		return "float16"
	case Float32:
		return "float32"
	case BFloat16:
		return "bfloat16"
	case Complex64:
		return "complex64"
	case Float64:
		return "float64"
	default:
		return "invalid"
	}
}

// Size returns the element size in bytes.
func (d Dtype) Size() int {
	switch d {
	case Bool, Uint8, Int8:
		return 1
	case Uint16, Int16, Float16, BFloat16:
		return 2
	case Uint32, Int32, Float32:
		return 4
	case Uint64, Int64, Float64, Complex64:
		return 8
	default:
		return 0
	}
}

// Valid reports whether d is a known dtype.
func (d Dtype) Valid() bool { return d >= Bool && d <= Float64 }

// elementCount returns the product of a shape, with the empty shape (a scalar)
// counting as one element.
func elementCount(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// errShape reports a mismatch between a host data length and a declared shape.
func errShape(op string, dataLen int, shape []int) error {
	return fmt.Errorf("mlxgo: %s: data length %d does not match shape %v (%d elements)",
		op, dataLen, shape, elementCount(shape))
}
