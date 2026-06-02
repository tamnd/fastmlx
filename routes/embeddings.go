// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// embeddingRequest is the OpenAI-compatible /v1/embeddings request. input is a
// single string or a list of strings; items is the structured multimodal form.
// Exactly one of the two must be present. The pointer on items distinguishes an
// absent field from a present-but-empty list, which the validation needs.
type embeddingRequest struct {
	Input          json.RawMessage    `json:"input"`
	Items          *[]json.RawMessage `json:"items"`
	Model          string             `json:"model"`
	EncodingFormat *string            `json:"encoding_format"`
	Dimensions     *int               `json:"dimensions"`
}

// Embeddings handles POST /v1/embeddings. It validates and normalizes the
// request exactly as the reference does, then the embedding computation is
// served by the embedding engine. That engine is part of the compute backend and
// lands in a later milestone, so the compute step reports as not implemented; the
// request validation/normalization and the byte-exact response assembly
// (api.BuildEmbeddingResponse, with base64 encoding and dimension truncation)
// are ported and exercised by parity fixtures so the wire form is locked in
// ahead of the engine.
func (rt *Router) Embeddings(w http.ResponseWriter, r *http.Request) {
	var req embeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusUnprocessableEntity, "model is required", "invalid_request_error", "model")
		return
	}
	if req.EncodingFormat != nil && *req.EncodingFormat != "float" && *req.EncodingFormat != "base64" {
		writeError(w, http.StatusUnprocessableEntity,
			"encoding_format must be 'float' or 'base64'", "invalid_request_error", "encoding_format")
		return
	}

	hasInput := len(req.Input) > 0 && string(req.Input) != "null"
	hasItems := req.Items != nil
	switch {
	case !hasInput && !hasItems:
		writeError(w, http.StatusUnprocessableEntity,
			"Either input or items must be provided", "invalid_request_error", "")
		return
	case hasInput && hasItems:
		writeError(w, http.StatusUnprocessableEntity,
			"input and items cannot be provided together", "invalid_request_error", "")
		return
	}

	var inputs []string
	if hasItems {
		if len(*req.Items) == 0 {
			writeError(w, http.StatusUnprocessableEntity, "items cannot be empty", "invalid_request_error", "items")
			return
		}
		if !validateEmbeddingItems(w, *req.Items) {
			return
		}
		inputs = make([]string, len(*req.Items)) // placeholder per item for the empty check
	} else {
		parsed, ok := parseEmbeddingInput(w, req.Input)
		if !ok {
			return
		}
		inputs = parsed
	}

	// Mirrors the reference handler's post-normalization guard.
	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, "Input cannot be empty", "invalid_request_error", "input")
		return
	}

	writeError(w, http.StatusNotImplemented,
		"embeddings are served by the embedding engine, which lands with the compute backend",
		"not_implemented_error", "")
}

// parseEmbeddingInput normalizes the input field to a list of strings. A single
// string becomes a one-element list; a list of strings passes through. Anything
// else is a 422, matching the pydantic Union[str, List[str]] type.
func parseEmbeddingInput(w http.ResponseWriter, raw json.RawMessage) ([]string, bool) {
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return []string{single}, true
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list, true
	}
	writeError(w, http.StatusUnprocessableEntity,
		"input must be a string or a list of strings", "invalid_request_error", "input")
	return nil, false
}

// validateEmbeddingItems checks each structured item the way the reference's
// EmbeddingInputItem model does: unknown fields are rejected (extra="forbid")
// and each item must carry text or image.
func validateEmbeddingItems(w http.ResponseWriter, items []json.RawMessage) bool {
	for _, raw := range items {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		var item struct {
			Text  *string `json:"text"`
			Image *string `json:"image"`
		}
		if err := dec.Decode(&item); err != nil {
			writeError(w, http.StatusUnprocessableEntity,
				"invalid embedding item: "+err.Error(), "invalid_request_error", "items")
			return false
		}
		if item.Text == nil && item.Image == nil {
			writeError(w, http.StatusUnprocessableEntity,
				"Embedding input item must include text or image", "invalid_request_error", "items")
			return false
		}
	}
	return true
}
