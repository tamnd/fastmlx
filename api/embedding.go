// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/binary"
	"math"
)

// This file ports the GPU-free half of the /v1/embeddings endpoint: input
// normalization, base64 encoding, dimension truncation with renormalization,
// and byte-exact response assembly. The embedding vectors themselves come from
// the embedding engine on the MLX backend, which is compute-gated and lands with
// the rest of the compute layer.

// EncodeEmbeddingBase64 encodes a vector the way OpenAI does: little-endian
// single-precision floats, then standard base64. It mirrors the reference's
// struct.pack("<{n}f", ...) + base64.b64encode.
func EncodeEmbeddingBase64(embedding []float64) string {
	buf := make([]byte, 4*len(embedding))
	for i, v := range embedding {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// TruncateEmbedding truncates a vector to dimensions and renormalizes it to unit
// L2 length, so cosine similarity is preserved. A request for at least the full
// width returns the vector unchanged. A zero-norm truncation is returned as-is.
func TruncateEmbedding(embedding []float64, dimensions int) []float64 {
	if dimensions >= len(embedding) {
		return embedding
	}
	truncated := embedding[:dimensions]
	var sumSq float64
	for _, x := range truncated {
		sumSq += x * x
	}
	norm := math.Sqrt(sumSq)
	if norm > 0 {
		out := make([]float64, len(truncated))
		for i, x := range truncated {
			out[i] = x / norm
		}
		return out
	}
	return truncated
}

// NormalizeEmbeddingInput turns the OpenAI "input" field (a single string or a
// list of strings) into a list of strings.
func NormalizeEmbeddingInput(single string, list []string, isList bool) []string {
	if isList {
		return append([]string(nil), list...)
	}
	return []string{single}
}

// EmbeddingItem is one structured multimodal embedding input.
type EmbeddingItem struct {
	Text  string
	Image string
}

// NormalizeEmbeddingItems reduces structured items to ordered text/image pairs,
// dropping unset fields, matching the reference normalize_embedding_items.
func NormalizeEmbeddingItems(items []EmbeddingItem) []EmbeddingItem {
	out := make([]EmbeddingItem, 0, len(items))
	for _, it := range items {
		out = append(out, EmbeddingItem{Text: it.Text, Image: it.Image})
	}
	return out
}

// BuildEmbeddingResponse assembles the /v1/embeddings response body. embeddings
// is the per-input vector list from the engine. When dimensions is set and
// smaller than a vector, that vector is truncated and renormalized. The
// encodingFormat selects a float array or a base64 string per embedding. The
// wire form is pydantic model_dump_json (compact, non-ASCII passthrough), so it
// routes through dumpCompact, byte-for-byte with the reference.
func BuildEmbeddingResponse(model string, embeddings [][]float64, totalTokens int, encodingFormat string, dimensions *int) string {
	data := jval{kind: kindArray}
	for i, emb := range embeddings {
		if dimensions != nil && *dimensions < len(emb) {
			emb = TruncateEmbedding(emb, *dimensions)
		}
		var embVal jval
		if encodingFormat == "base64" {
			embVal = jstr(EncodeEmbeddingBase64(emb))
		} else {
			arr := jval{kind: kindArray}
			for _, f := range emb {
				arr.arr = append(arr.arr, jfloat(f))
			}
			embVal = arr
		}
		data.arr = append(data.arr, jobj(
			"object", jstr("embedding"),
			"index", jint(i),
			"embedding", embVal,
		))
	}
	resp := jobj(
		"object", jstr("list"),
		"data", data,
		"model", jstr(model),
		"usage", jobj(
			"prompt_tokens", jint(totalTokens),
			"total_tokens", jint(totalTokens),
		),
	)
	return resp.dumpCompact()
}
