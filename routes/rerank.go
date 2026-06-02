// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"

	"github.com/tamnd/fastmlx/api"
)

// rerankRequest is the Cohere/Jina-compatible /v1/rerank request. query is a
// string or an object (multimodal rerankers); documents is a list of strings or
// objects. top_n limits the result count; return_documents controls whether the
// documents are echoed back. The raw forms are kept so the response assembly can
// echo objects through unchanged.
type rerankRequest struct {
	Model           string            `json:"model"`
	Query           json.RawMessage   `json:"query"`
	Documents       []json.RawMessage `json:"documents"`
	TopN            *int              `json:"top_n"`
	ReturnDocuments *bool             `json:"return_documents"`
	MaxChunksPerDoc *int              `json:"max_chunks_per_doc"`
}

// Rerank handles POST /v1/rerank. It validates the request the same way the
// reference does (non-empty documents, non-empty query) and then performs the
// relevance scoring, which is served by the reranker engine. That engine is part
// of the compute backend and lands in a later milestone, so the scoring step
// reports as not implemented; the request validation, normalization, and
// response assembly (BuildRerankResponse) are ported and exercised by parity
// fixtures so the wire form is locked in ahead of the engine.
func (rt *Router) Rerank(w http.ResponseWriter, r *http.Request) {
	var req rerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}

	docsText := api.NormalizeDocuments(req.Documents)
	if len(docsText) == 0 {
		writeError(w, http.StatusBadRequest, "Documents cannot be empty", "invalid_request_error", "documents")
		return
	}
	if !queryPresent(req.Query) {
		writeError(w, http.StatusBadRequest, "Query cannot be empty", "invalid_request_error", "query")
		return
	}

	writeError(w, http.StatusNotImplemented,
		"reranking is served by the reranker engine, which lands with the compute backend",
		"not_implemented_error", "")
}

// queryPresent reports whether the query is a non-empty string or a non-empty
// object, matching the reference's `if not request.query` truthiness check
// (an empty string or empty dict is falsy).
func queryPresent(raw json.RawMessage) bool {
	if len(raw) == 0 || string(raw) == "null" {
		return false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s != ""
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		return len(obj) > 0
	}
	return true
}
