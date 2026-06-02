// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"sort"
	"sync"

	"github.com/tamnd/fastmlx/tokenizer"
)

// DefaultMockResponse is the canned text the mock backend emits, tokenized with
// the request's tokenizer. It round-trips to readable text through the mock
// tokenizer so OpenAI SDK clients see a coherent (if fixed) completion.
const DefaultMockResponse = "This is a mock response from the fastmlx serving layer. " +
	"The compute backend is not loaded yet, so these tokens are produced deterministically " +
	"to exercise scheduling, streaming, and the OpenAI-compatible API end to end."

// MockDecode is a deterministic DecodeStrategy that emits a canned token stream,
// one token per active sequence per step. It carries no model and no GPU, so the
// scheduler, HTTP routes, SSE encoder, and benchmarks all run before the real
// compute backend exists. It satisfies pipeline.DecodeStrategy so the real
// BatchGenerator drops in with zero scheduler changes.
type MockDecode struct {
	tok      tokenizer.Tokenizer
	response []int // canned output tokens, planned once

	mu      sync.Mutex
	nextUID int
	active  map[int]*mockSeq
}

type mockSeq struct {
	pos       int
	genLen    int    // number of tokens this sequence will emit
	endReason string // finish reason reported on the final token
}

// NewMockDecode builds a mock backend that emits the given response text (empty
// uses DefaultMockResponse), tokenized with tok.
func NewMockDecode(tok tokenizer.Tokenizer, response string) *MockDecode {
	if response == "" {
		response = DefaultMockResponse
	}
	return &MockDecode{
		tok:      tok,
		response: tok.Encode(response),
		active:   make(map[int]*mockSeq),
	}
}

// Insert admits a sequence and plans how many canned tokens it will emit, capped
// by MaxTokens.
func (m *MockDecode) Insert(req DecodeRequest) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	genLen := len(m.response)
	endReason := "stop"
	if req.MaxTokens > 0 && req.MaxTokens < genLen {
		genLen = req.MaxTokens
		endReason = "length"
	}
	uid := m.nextUID
	m.nextUID++
	m.active[uid] = &mockSeq{genLen: genLen, endReason: endReason}
	return uid, nil
}

// Step advances every active sequence by one token. UIDs are visited in a stable
// order so output is deterministic.
func (m *MockDecode) Step() ([]TokenResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.active) == 0 {
		return nil, nil
	}
	uids := make([]int, 0, len(m.active))
	for uid := range m.active {
		uids = append(uids, uid)
	}
	sort.Ints(uids)

	results := make([]TokenResult, 0, len(uids))
	for _, uid := range uids {
		seq := m.active[uid]
		var tok int
		if seq.genLen == 0 {
			// Zero-length generation (MaxTokens 0): emit EOS and finish.
			tok = m.tok.EOSTokenID()
		} else {
			tok = m.response[seq.pos%len(m.response)]
			seq.pos++
		}
		res := TokenResult{UID: uid, Token: tok}
		if seq.pos >= seq.genLen {
			res.FinishReason = seq.endReason
		}
		results = append(results, res)
	}
	return results, nil
}

// Remove retires a sequence. The mock holds no KV cache, so it returns nil.
func (m *MockDecode) Remove(uid int) any {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.active, uid)
	return nil
}

// HasActive reports whether any sequence is still decoding.
func (m *MockDecode) HasActive() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.active) > 0
}

// Close releases resources. The mock has none.
func (m *MockDecode) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.active = make(map[int]*mockSeq)
	return nil
}

// compile-time check.
var _ DecodeStrategy = (*MockDecode)(nil)
