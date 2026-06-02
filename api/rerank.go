// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// Rerank (Cohere/Jina-compatible /v1/rerank) request and response shapes.
//
// The relevance scoring itself is compute-gated: it needs a sequence-
// classification reranker model loaded on the MLX backend, so it lands with the
// compute layer. Everything around it is pure and ported here: input
// normalization, the empty-query/empty-documents guards, and the response
// assembly (ranked-and-sliced index order, return_documents handling, the
// rerank-<hex> id, and the usage block). The wire form matches the reference's
// FastAPI JSON exactly (compact separators, non-ASCII passed through), the same
// encoding dumpCompact produces.

// NewRerankID mints a response id of the form rerank-<8 hex>, matching the
// reference's uuid4().hex[:8] default.
func NewRerankID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return "rerank-" + hex.EncodeToString(b[:])
}

// NormalizeDocuments reduces each raw document (a JSON string or object) to its
// text view, used only for the empty-documents guard. A string becomes itself;
// an object becomes its "text" field (empty when absent); anything else becomes
// its JSON text. This mirrors the reference normalize_documents.
func NormalizeDocuments(documents []json.RawMessage) []string {
	out := make([]string, 0, len(documents))
	for _, raw := range documents {
		v, ok := parseOrdered(string(raw))
		if !ok {
			out = append(out, strings.TrimSpace(string(raw)))
			continue
		}
		switch v.kind {
		case kindString:
			out = append(out, v.s)
		case kindObject:
			out = append(out, v.getString("text"))
		default:
			out = append(out, v.dumpCompact())
		}
	}
	return out
}

// BuildRerankResponse assembles the rerank response body. scores is indexed by
// the original document position; order is the already-ranked, already-top_n-
// sliced index list returned by the reranker engine. documents are the raw input
// values (strings or objects); when returnDocuments is set each result carries
// its document (a string wraps into {"text": ...}, an object passes through
// as-is), otherwise document is null.
func BuildRerankResponse(id, model string, scores []float64, order []int, documents []json.RawMessage, returnDocuments bool, totalTokens int) string {
	results := jval{kind: kindArray}
	for _, idx := range order {
		var doc jval
		if returnDocuments {
			doc = rerankDocument(documents, idx)
		} else {
			doc = jnull()
		}
		var score float64
		if idx >= 0 && idx < len(scores) {
			score = scores[idx]
		}
		results.arr = append(results.arr, jobj(
			"index", jint(idx),
			"relevance_score", jfloat(score),
			"document", doc,
		))
	}
	resp := jobj(
		"id", jstr(id),
		"results", results,
		"model", jstr(model),
		"usage", jobj("total_tokens", jint(totalTokens)),
	)
	return resp.dumpCompact()
}

// rerankDocument renders the document at idx for the response: an object input
// passes through unchanged, any other input (a string) wraps into {"text": ...}.
func rerankDocument(documents []json.RawMessage, idx int) jval {
	if idx < 0 || idx >= len(documents) {
		return jobj("text", jstr(""))
	}
	v, ok := parseOrdered(string(documents[idx]))
	if ok && v.kind == kindObject {
		return v
	}
	if ok && v.kind == kindString {
		return jobj("text", jstr(v.s))
	}
	return jobj("text", jstr(strings.TrimSpace(string(documents[idx]))))
}

// jfloat builds a number value formatted the way Python's json.dumps renders a
// float: the shortest round-tripping decimal, always with a fractional part or
// exponent so it reads back as a float (1.0, not 1).
func jfloat(f float64) jval {
	return jval{kind: kindNumber, s: formatPyFloat(f)}
}

// formatPyFloat reproduces Python's repr(float)/json.dumps formatting. Python
// uses fixed notation when the value is zero or its magnitude is in
// [1e-4, 1e16), and scientific notation otherwise, always choosing the shortest
// string that round-trips. Relevance scores live in [0, 1], so the fixed branch
// is the one that matters; the scientific branch is kept faithful for the tail.
func formatPyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}
	abs := math.Abs(f)
	if f == 0 || (abs >= 1e-4 && abs < 1e16) {
		s := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.ContainsRune(s, '.') {
			s += ".0"
		}
		return s
	}
	return strconv.FormatFloat(f, 'e', -1, 64)
}
