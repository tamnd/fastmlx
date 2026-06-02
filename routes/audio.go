// SPDX-License-Identifier: MIT OR Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/tamnd/fastmlx/api"
)

// This file wires the three OpenAI-compatible audio endpoints. The request
// validation mirrors the reference exactly; the speech-to-text, text-to-speech,
// and speech-to-speech computation runs on the STT/TTS/STS engines on the MLX
// backend, which are compute-gated and land with the rest of the compute layer,
// so a valid request reports not-implemented after passing validation.

// writeAudioError maps an api.AudioError onto the OpenAI error envelope.
func writeAudioError(w http.ResponseWriter, e *api.AudioError) {
	typ := "invalid_request_error"
	if e.Status >= 500 {
		typ = "internal_error"
	}
	writeError(w, e.Status, e.Message, typ, "")
}

// Transcribe handles POST /v1/audio/transcriptions. It is a multipart upload
// with a required audio file and model. response_format and temperature are
// accepted for OpenAI compatibility but ignored, matching the reference.
func (rt *Router) Transcribe(w http.ResponseWriter, r *http.Request) {
	if !rt.parseAudioUpload(w, r) {
		return
	}
	writeError(w, http.StatusNotImplemented,
		"transcription is served by the speech-to-text engine, which lands with the compute backend",
		"not_implemented_error", "")
}

// AudioProcess handles POST /v1/audio/process (speech enhancement / source
// separation / speech-to-speech). Same multipart shape as transcription.
func (rt *Router) AudioProcess(w http.ResponseWriter, r *http.Request) {
	if !rt.parseAudioUpload(w, r) {
		return
	}
	writeError(w, http.StatusNotImplemented,
		"audio processing is served by the speech-to-speech engine, which lands with the compute backend",
		"not_implemented_error", "")
}

// parseAudioUpload validates the multipart upload shared by the transcription
// and process endpoints: a required "file" part and a required "model" form
// field, with the upload capped at the reference's limit. It returns false and
// writes the error response when validation fails.
func (rt *Router) parseAudioUpload(w http.ResponseWriter, r *http.Request) bool {
	if err := r.ParseMultipartForm(api.MaxAudioUploadBytes); err != nil {
		writeError(w, http.StatusUnprocessableEntity,
			"request must be multipart/form-data with a file and model", "invalid_request_error", "")
		return false
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "file is required", "invalid_request_error", "file")
		return false
	}
	defer file.Close()
	if header.Size > api.MaxAudioUploadBytes {
		writeAudioError(w, &api.AudioError{
			Status:  413,
			Message: "Audio file exceeds maximum allowed size (104857600 bytes)",
		})
		return false
	}
	if r.FormValue("model") == "" {
		writeError(w, http.StatusUnprocessableEntity, "model is required", "invalid_request_error", "model")
		return false
	}
	return true
}

// speechRequest is the /v1/audio/speech JSON body. The pointer fields carry the
// optional parameters whose presence (not just value) the validation needs.
type speechRequest struct {
	Model             string   `json:"model"`
	Input             string   `json:"input"`
	Voice             *string  `json:"voice"`
	Instructions      *string  `json:"instructions"`
	Speed             *float64 `json:"speed"`
	ResponseFormat    *string  `json:"response_format"`
	RefAudio          *string  `json:"ref_audio"`
	RefText           *string  `json:"ref_text"`
	Temperature       *float64 `json:"temperature"`
	TopK              *int     `json:"top_k"`
	TopP              *float64 `json:"top_p"`
	RepetitionPenalty *float64 `json:"repetition_penalty"`
	MaxTokens         *int     `json:"max_tokens"`
	Stream            bool     `json:"stream"`
	StreamingInterval *float64 `json:"streaming_interval"`
}

// Speech handles POST /v1/audio/speech (text-to-speech). It validates the JSON
// body the way the reference create_speech does: a non-empty input, the
// streaming response-format restriction, the streaming-interval bounds, and the
// ref-audio base64 decode, then reports the synthesis step as not implemented.
func (rt *Router) Speech(w http.ResponseWriter, r *http.Request) {
	var req speechRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "")
		return
	}

	if strings.TrimSpace(req.Input) == "" {
		writeError(w, http.StatusBadRequest, "'input' field must not be empty", "invalid_request_error", "input")
		return
	}

	if req.Stream {
		// Default response_format is "wav"; only an explicit non-wav format is
		// rejected for the streaming path.
		if req.ResponseFormat != nil && *req.ResponseFormat != "wav" {
			writeError(w, http.StatusBadRequest,
				"Streaming TTS currently only supports response_format='wav'", "invalid_request_error", "response_format")
			return
		}
		if _, e := api.ResolveTTSStreamingInterval(req.StreamingInterval); e != nil {
			writeAudioError(w, e)
			return
		}
	}

	refText := ""
	if req.RefText != nil {
		refText = *req.RefText
	}
	if _, e := api.DecodeRefAudioBase64(req.RefAudio, refText); e != nil {
		writeAudioError(w, e)
		return
	}

	writeError(w, http.StatusNotImplemented,
		"speech synthesis is served by the text-to-speech engine, which lands with the compute backend",
		"not_implemented_error", "")
}
