// SPDX-License-Identifier: MIT OR Apache-2.0

package eval

import "math"

// PyRandom reproduces the part of Python's random.Random that the dataset
// samplers depend on: a Mersenne Twister (MT19937) seeded the way CPython seeds
// from an integer, plus getrandbits, _randbelow, and sample. Reproducing the
// exact pseudo-random stream is what lets the Go samplers pick the same question
// subset for a given seed as the reference, so two models are scored on
// identical questions. Only the integer-seed and small-bit-width paths the
// samplers actually use are implemented.

const mtN = 624

type mtState struct {
	mt  [mtN]uint32
	mti int
}

func (s *mtState) initGenrand(seed uint32) {
	s.mt[0] = seed
	for i := 1; i < mtN; i++ {
		prev := s.mt[i-1]
		s.mt[i] = 1812433253*(prev^(prev>>30)) + uint32(i)
	}
	s.mti = mtN
}

// initByArray mirrors CPython's init_by_array, which is how random.seed(int)
// initializes the generator after splitting the integer into 32-bit words.
func (s *mtState) initByArray(key []uint32) {
	s.initGenrand(19650218)
	i, j := 1, 0
	k := max(mtN, len(key))
	for ; k > 0; k-- {
		prev := s.mt[i-1]
		s.mt[i] = (s.mt[i] ^ ((prev ^ (prev >> 30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= mtN {
			s.mt[0] = s.mt[mtN-1]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k = mtN - 1; k > 0; k-- {
		prev := s.mt[i-1]
		s.mt[i] = (s.mt[i] ^ ((prev ^ (prev >> 30)) * 1566083941)) - uint32(i)
		i++
		if i >= mtN {
			s.mt[0] = s.mt[mtN-1]
			i = 1
		}
	}
	s.mt[0] = 0x80000000
}

func (s *mtState) genrandUint32() uint32 {
	const (
		m         = 397
		matrixA   = 0x9908b0df
		upperMask = 0x80000000
		lowerMask = 0x7fffffff
	)
	if s.mti >= mtN {
		var y uint32
		for kk := range mtN - m {
			y = (s.mt[kk] & upperMask) | (s.mt[kk+1] & lowerMask)
			s.mt[kk] = s.mt[kk+m] ^ (y >> 1) ^ ((y & 1) * matrixA)
		}
		for kk := mtN - m; kk < mtN-1; kk++ {
			y = (s.mt[kk] & upperMask) | (s.mt[kk+1] & lowerMask)
			s.mt[kk] = s.mt[kk+(m-mtN)] ^ (y >> 1) ^ ((y & 1) * matrixA)
		}
		y = (s.mt[mtN-1] & upperMask) | (s.mt[0] & lowerMask)
		s.mt[mtN-1] = s.mt[m-1] ^ (y >> 1) ^ ((y & 1) * matrixA)
		s.mti = 0
	}
	y := s.mt[s.mti]
	s.mti++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// PyRandom is a Python-compatible random source seeded from an integer.
type PyRandom struct {
	st mtState
}

// NewPyRandom seeds the generator the way CPython's random.Random(seed) does for
// a non-negative integer: the seed is split into little-endian 32-bit words
// (with at least one word) and fed to init_by_array.
func NewPyRandom(seed uint64) *PyRandom {
	var key []uint32
	if seed == 0 {
		key = []uint32{0}
	} else {
		for seed > 0 {
			key = append(key, uint32(seed&0xffffffff))
			seed >>= 32
		}
	}
	r := &PyRandom{}
	r.st.initByArray(key)
	return r
}

// GetRandBits returns k random bits, matching Python's getrandbits for the
// single-word range 1 <= k <= 32 that the samplers exercise.
func (r *PyRandom) GetRandBits(k int) uint32 {
	if k <= 0 {
		return 0
	}
	if k > 32 {
		k = 32
	}
	return r.st.genrandUint32() >> (32 - uint(k))
}

// RandBelow returns a uniform integer in [0, n), matching CPython's
// _randbelow_with_getrandbits: it draws bit_length(n) bits and rejects values at
// or above n. Returns 0 when n is 0.
func (r *PyRandom) RandBelow(n int) int {
	if n <= 0 {
		return 0
	}
	k := bitLength(n)
	v := int(r.GetRandBits(k))
	for v >= n {
		v = int(r.GetRandBits(k))
	}
	return v
}

func bitLength(n int) int {
	bits := 0
	for n > 0 {
		bits++
		n >>= 1
	}
	return bits
}

// ceilLog4 returns ceil(log_4(m)), the exponent CPython's sample uses to size
// the selection set. math.ceil yields an int there, so the result is an exact
// integer exponent.
func ceilLog4(m int) int {
	return int(math.Ceil(math.Log(float64(m)) / math.Log(4)))
}

// pow4 returns 4**e as an exact integer (e is non-negative here).
func pow4(e int) int {
	return 1 << (2 * e)
}

// SampleIndices returns k distinct indices into a population of size n in
// selection order, reproducing CPython's Random.sample index choices (the
// pool-versus-set branch and its randbelow call pattern) so the same elements
// are picked for a given seed.
func (r *PyRandom) SampleIndices(n, k int) []int {
	result := make([]int, k)
	setsize := 21
	if k > 5 {
		setsize += pow4(ceilLog4(k * 3))
	}
	if n <= setsize {
		pool := make([]int, n)
		for i := range pool {
			pool[i] = i
		}
		for i := range k {
			j := r.RandBelow(n - i)
			result[i] = pool[j]
			pool[j] = pool[n-i-1]
		}
	} else {
		selected := make(map[int]struct{}, k)
		for i := range k {
			j := r.RandBelow(n)
			for _, ok := selected[j]; ok; _, ok = selected[j] {
				j = r.RandBelow(n)
			}
			selected[j] = struct{}{}
			result[i] = j
		}
	}
	return result
}
