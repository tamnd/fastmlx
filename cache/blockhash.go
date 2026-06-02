// SPDX-License-Identifier: MIT OR Apache-2.0

// Package cache ports the GPU-free half of the block-aware prefix cache: the
// content-addressed block hashing and the shared-prefix matching the scheduler
// uses to reuse a previously computed KV prefix. The KV tensors themselves, the
// paged hot store, and the SSD cold store are compute/IO-gated and land with the
// backend; the hashing and matching here are pure functions of the token stream
// and are exercised by deterministic fixtures.
//
// The hash chain is seeded with a fastmlx-specific root, so fastmlx blocks form
// their own self-consistent namespace (not byte-cross-compatible with other
// engines' caches by design).
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
)

// DefaultBlockSize is the token count per cache block.
const DefaultBlockSize = 64

// rootSeed seeds the hash of the first block in a chain, so a block's hash
// depends on the whole prefix before it (vLLM-style chaining).
const rootSeed = "fastmlx-root"

// BlockHash is a content-addressed block identifier: the SHA-256 digest over
// the model name, the parent block's hash, and this block's tokens. It is
// comparable, so it works directly as a map key.
type BlockHash [sha256.Size]byte

// String renders the hash as lowercase hex.
func (h BlockHash) String() string { return hex.EncodeToString(h[:]) }

// tupleRepr renders a token slice as a Python tuple literal, matching the form
// the hash content is built from: "()" for empty, "(5,)" for one element,
// "(1, 2, 3)" for several.
func tupleRepr(tokenIDs []int) string {
	if len(tokenIDs) == 0 {
		return "()"
	}
	var b strings.Builder
	b.WriteByte('(')
	for i, v := range tokenIDs {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(v))
	}
	if len(tokenIDs) == 1 {
		b.WriteByte(',')
	}
	b.WriteByte(')')
	return b.String()
}

// ComputeBlockHash hashes one block from its parent hash (nil for the first
// block in a chain), its tokens, and the model name (empty to skip, used to
// isolate caches between models). Chaining each block's hash onto its parent is
// what makes a prefix match imply every earlier block matched too.
func ComputeBlockHash(parent *BlockHash, tokenIDs []int, modelName string) BlockHash {
	h := sha256.New()
	if modelName != "" {
		h.Write([]byte(modelName))
	}
	if parent != nil {
		h.Write(parent[:])
	} else {
		h.Write([]byte(rootSeed))
	}
	h.Write([]byte(tupleRepr(tokenIDs)))
	var out BlockHash
	copy(out[:], h.Sum(nil))
	return out
}

// HashFullBlocks chains the block hashes for every full block in tokenIDs. A
// trailing partial block (fewer than blockSize tokens) is not hashed, matching
// the reference num_full_blocks = len // block_size. A non-positive blockSize
// yields no hashes.
func HashFullBlocks(tokenIDs []int, blockSize int, modelName string) []BlockHash {
	if blockSize <= 0 {
		return nil
	}
	numFull := len(tokenIDs) / blockSize
	hashes := make([]BlockHash, 0, numFull)
	var parent *BlockHash
	for i := range numFull {
		block := tokenIDs[i*blockSize : i*blockSize+blockSize]
		h := ComputeBlockHash(parent, block, modelName)
		hashes = append(hashes, h)
		p := h
		parent = &p
	}
	return hashes
}

// FindSharedPrefix walks the full blocks of tokenIDs, hashing each onto its
// parent, and returns the longest run of leading blocks that the cache already
// holds (per the has membership check), the count of tokens those blocks cover,
// and the still-uncomputed tail. It stops at the first miss: because the hashes
// chain, a miss means no later block can match either. This mirrors the
// reference get_computed_blocks / find_shared_prefix.
func FindSharedPrefix(tokenIDs []int, blockSize int, modelName string, has func(BlockHash) bool) (shared []BlockHash, numCachedTokens int, remaining []int) {
	if blockSize > 0 {
		numFull := len(tokenIDs) / blockSize
		var parent *BlockHash
		for i := range numFull {
			block := tokenIDs[i*blockSize : i*blockSize+blockSize]
			h := ComputeBlockHash(parent, block, modelName)
			if has == nil || !has(h) {
				break
			}
			shared = append(shared, h)
			numCachedTokens += blockSize
			p := h
			parent = &p
		}
	}
	remaining = tokenIDs[numCachedTokens:]
	return shared, numCachedTokens, remaining
}
