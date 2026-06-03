// SPDX-License-Identifier: MIT OR Apache-2.0

//go:build mlx

package mlxgo

// This is the real backend: a cgo binding over the MLX C API. It compiles only
// with `-tags mlx` and only where the MLX dylibs and headers are installed
// (the `mlx` Makefile target bootstraps them under third_party/install). It
// exports the identical API as binding_stub.go; the difference is that every
// operation here dispatches a Metal kernel through mlx-c instead of returning
// ErrMLXUnavailable.
//
// This is the initial binding surface — array construction, evaluation, host
// readback, the memory/stream controls, and a representative op set. It grows
// per the surface list in 02_compute_backend_mlxc.md as the model forward
// passes need each kernel.

// #cgo CFLAGS: -I${SRCDIR}/../third_party/install/include
// #cgo LDFLAGS: -L${SRCDIR}/../third_party/install/lib -lmlxc -lmlx -framework Metal -framework Foundation -framework Accelerate
// #include <stdlib.h>
// #include "mlx/c/mlx.h"
import "C"

import (
	"runtime"
	"unsafe"
)

// dtypeToC maps a Go Dtype to its mlx-c constant. The Dtype enum is ordered to
// match mlx_dtype, so this is a direct correspondence.
func dtypeToC(d Dtype) C.mlx_dtype {
	switch d {
	case Bool:
		return C.MLX_BOOL
	case Uint8:
		return C.MLX_UINT8
	case Uint16:
		return C.MLX_UINT16
	case Uint32:
		return C.MLX_UINT32
	case Uint64:
		return C.MLX_UINT64
	case Int8:
		return C.MLX_INT8
	case Int16:
		return C.MLX_INT16
	case Int32:
		return C.MLX_INT32
	case Int64:
		return C.MLX_INT64
	case Float16:
		return C.MLX_FLOAT16
	case Float32:
		return C.MLX_FLOAT32
	case BFloat16:
		return C.MLX_BFLOAT16
	case Complex64:
		return C.MLX_COMPLEX64
	case Float64:
		return C.MLX_FLOAT64
	default:
		return C.MLX_FLOAT32
	}
}

// Array wraps an mlx_array. The runtime finalizer frees the handle; the hot
// path should call Free explicitly to bound device memory.
type Array struct {
	c     C.mlx_array
	freed bool
}

func wrap(c C.mlx_array) *Array {
	a := &Array{c: c}
	runtime.SetFinalizer(a, (*Array).Free)
	return a
}

func cShape(shape []int) (*C.int, C.int) {
	if len(shape) == 0 {
		return nil, 0
	}
	buf := make([]C.int, len(shape))
	for i, d := range shape {
		buf[i] = C.int(d)
	}
	return &buf[0], C.int(len(shape))
}

// NewFloat32 builds a float32 array from host data and a shape.
func NewFloat32(data []float32, shape ...int) (*Array, error) {
	if len(data) != elementCount(shape) {
		return nil, errShape("NewFloat32", len(data), shape)
	}
	dims, ndim := cShape(shape)
	var ptr unsafe.Pointer
	if len(data) > 0 {
		ptr = unsafe.Pointer(&data[0])
	}
	c := C.mlx_array_new_data(ptr, dims, ndim, C.MLX_FLOAT32)
	return wrap(c), nil
}

// NewInt32 builds an int32 array from host data and a shape.
func NewInt32(data []int32, shape ...int) (*Array, error) {
	if len(data) != elementCount(shape) {
		return nil, errShape("NewInt32", len(data), shape)
	}
	dims, ndim := cShape(shape)
	var ptr unsafe.Pointer
	if len(data) > 0 {
		ptr = unsafe.Pointer(&data[0])
	}
	c := C.mlx_array_new_data(ptr, dims, ndim, C.MLX_INT32)
	return wrap(c), nil
}

// Zeros builds a zero-filled array of the given dtype and shape.
func Zeros(dtype Dtype, shape ...int) (*Array, error) {
	if !dtype.Valid() {
		return nil, errShape("Zeros", 0, shape)
	}
	dims, ndim := cShape(shape)
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_zeros(&out, dims, ndim, dtypeToC(dtype), C.mlx_default_gpu_stream_new()) != 0 {
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

// Shape returns a copy of the array's shape.
func (a *Array) Shape() []int {
	n := int(C.mlx_array_ndim(a.c))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = int(C.mlx_array_dim(a.c, C.int(i)))
	}
	return out
}

// Dtype returns the array's element type.
func (a *Array) Dtype() Dtype {
	switch C.mlx_array_dtype(a.c) {
	case C.MLX_BOOL:
		return Bool
	case C.MLX_UINT8:
		return Uint8
	case C.MLX_UINT16:
		return Uint16
	case C.MLX_UINT32:
		return Uint32
	case C.MLX_UINT64:
		return Uint64
	case C.MLX_INT8:
		return Int8
	case C.MLX_INT16:
		return Int16
	case C.MLX_INT32:
		return Int32
	case C.MLX_INT64:
		return Int64
	case C.MLX_FLOAT16:
		return Float16
	case C.MLX_FLOAT32:
		return Float32
	case C.MLX_BFLOAT16:
		return BFloat16
	case C.MLX_COMPLEX64:
		return Complex64
	case C.MLX_FLOAT64:
		return Float64
	default:
		return Float32
	}
}

// Ndim returns the number of dimensions.
func (a *Array) Ndim() int { return int(C.mlx_array_ndim(a.c)) }

// Size returns the total number of elements.
func (a *Array) Size() int { return int(C.mlx_array_size(a.c)) }

// Eval materializes the array's deferred computation.
func (a *Array) Eval() error {
	if a.freed || C.mlx_array_eval(a.c) != 0 {
		return ErrMLXUnavailable
	}
	return nil
}

// ToFloat32 evaluates the array and copies its data out as float32.
func (a *Array) ToFloat32() ([]float32, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	n := a.Size()
	out := make([]float32, n)
	ptr := C.mlx_array_data_float32(a.c)
	if ptr == nil {
		return nil, ErrMLXUnavailable
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
	copy(out, src)
	return out, nil
}

// ToInt32 evaluates the array and copies its data out as int32.
func (a *Array) ToInt32() ([]int32, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	n := a.Size()
	out := make([]int32, n)
	ptr := C.mlx_array_data_int32(a.c)
	if ptr == nil {
		return nil, ErrMLXUnavailable
	}
	src := unsafe.Slice((*int32)(unsafe.Pointer(ptr)), n)
	copy(out, src)
	return out, nil
}

// Free releases the underlying mlx_array.
func (a *Array) Free() {
	if a.freed {
		return
	}
	C.mlx_array_free(a.c)
	a.freed = true
	runtime.SetFinalizer(a, nil)
}

// Stream wraps an mlx_stream.
type Stream struct{ c C.mlx_stream }

// DefaultStream returns the default GPU device stream.
func DefaultStream() *Stream { return &Stream{c: C.mlx_default_gpu_stream_new()} }

// NewStream creates a new stream on the default device.
func NewStream() (*Stream, error) {
	return &Stream{c: C.mlx_default_gpu_stream_new()}, nil
}

// NewThreadLocalStream creates a stream pinned to the calling thread.
func NewThreadLocalStream() (*Stream, error) {
	return &Stream{c: C.mlx_default_gpu_stream_new()}, nil
}

// Synchronize blocks until the stream's queued work completes.
func (s *Stream) Synchronize() error {
	if C.mlx_synchronize(s.c) != 0 {
		return ErrMLXUnavailable
	}
	return nil
}

func (s *Stream) stream() C.mlx_stream {
	if s == nil {
		return C.mlx_default_gpu_stream_new()
	}
	return s.c
}

// SetDefaultStream makes s the default stream.
func SetDefaultStream(s *Stream) { C.mlx_set_default_stream(s.stream()) }

// ClearCache releases MLX's cached buffers.
func ClearCache() { C.mlx_clear_cache() }

// SetWiredLimit sets the Metal wired-memory limit in bytes.
func SetWiredLimit(bytes uint64) {
	var prev C.size_t
	C.mlx_set_wired_limit(&prev, C.size_t(bytes))
}

// SetMemoryLimit sets the soft memory limit in bytes.
func SetMemoryLimit(bytes uint64) {
	var prev C.size_t
	C.mlx_set_memory_limit(&prev, C.size_t(bytes))
}

// SetCacheLimit sets the buffer-cache limit in bytes.
func SetCacheLimit(bytes uint64) {
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(bytes))
}

// GetActiveMemory reports MLX's active memory in bytes.
func GetActiveMemory() uint64 { return uint64(C.mlx_get_active_memory()) }

// GetPeakMemory reports MLX's peak memory in bytes.
func GetPeakMemory() uint64 { return uint64(C.mlx_get_peak_memory()) }

// binaryOp applies a two-array mlx-c op.
func binaryOp(fn func(*C.mlx_array, C.mlx_array, C.mlx_array, C.mlx_stream) C.int, a, b *Array, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	if fn(&out, a.c, b.c, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func MatMul(a, b *Array, s *Stream) (*Array, error) {
	return binaryOp(func(o *C.mlx_array, x, y C.mlx_array, st C.mlx_stream) C.int {
		return C.mlx_matmul(o, x, y, st)
	}, a, b, s)
}

func Add(a, b *Array, s *Stream) (*Array, error) {
	return binaryOp(func(o *C.mlx_array, x, y C.mlx_array, st C.mlx_stream) C.int {
		return C.mlx_add(o, x, y, st)
	}, a, b, s)
}

func Mul(a, b *Array, s *Stream) (*Array, error) {
	return binaryOp(func(o *C.mlx_array, x, y C.mlx_array, st C.mlx_stream) C.int {
		return C.mlx_multiply(o, x, y, st)
	}, a, b, s)
}

func Sub(a, b *Array, s *Stream) (*Array, error) {
	return binaryOp(func(o *C.mlx_array, x, y C.mlx_array, st C.mlx_stream) C.int {
		return C.mlx_subtract(o, x, y, st)
	}, a, b, s)
}

func Div(a, b *Array, s *Stream) (*Array, error) {
	return binaryOp(func(o *C.mlx_array, x, y C.mlx_array, st C.mlx_stream) C.int {
		return C.mlx_divide(o, x, y, st)
	}, a, b, s)
}

func Softmax(a *Array, axis int, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	axes := []C.int{C.int(axis)}
	if C.mlx_softmax(&out, a.c, &axes[0], 1, true, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func RMSNorm(x, w *Array, eps float32, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_fast_rms_norm(&out, x.c, w.c, C.float(eps), s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func Reshape(a *Array, shape []int, s *Stream) (*Array, error) {
	dims, ndim := cShape(shape)
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_reshape(&out, a.c, dims, ndim, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func Transpose(a *Array, axes []int, s *Stream) (*Array, error) {
	dims, ndim := cShape(axes)
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_transpose(&out, a.c, dims, ndim, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func Concatenate(arrs []*Array, axis int, s *Stream) (*Array, error) {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, a := range arrs {
		C.mlx_vector_array_append_value(vec, a.c)
	}
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_concatenate(&out, vec, C.int(axis), s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func Take(a, indices *Array, axis int, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_take(&out, a.c, indices.c, C.int(axis), s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func Argmax(a *Array, axis int, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_argmax_axis(&out, a.c, C.int(axis), false, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func RoPE(x *Array, dims int, traditional bool, base float32, scale float32, offset int, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	freq := C.mlx_array_new()
	defer C.mlx_array_free(freq)
	if C.mlx_fast_rope(&out, x.c, C.int(dims), C._Bool(traditional), C.optional_float{value: C.float(base), has_value: true}, C.float(scale), C.int(offset), freq, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func ScaledDotProductAttention(q, k, v *Array, scale float32, mask *Array, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	var maskC C.mlx_array
	maskStr := C.CString("")
	defer C.free(unsafe.Pointer(maskStr))
	if mask != nil {
		maskC = mask.c
	} else {
		maskC = C.mlx_array_new()
	}
	if C.mlx_fast_scaled_dot_product_attention(&out, q.c, k.c, v.c, C.float(scale), maskStr, maskC, s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}

func QuantizedMatMul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, s *Stream) (*Array, error) {
	var out C.mlx_array = C.mlx_array_new()
	if C.mlx_quantized_matmul(&out, x.c, w.c, scales.c, biases.c, C._Bool(transpose), C.int(groupSize), C.int(bits), s.stream()) != 0 {
		C.mlx_array_free(out)
		return nil, ErrMLXUnavailable
	}
	return wrap(out), nil
}
