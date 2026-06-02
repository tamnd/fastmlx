// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
)

// This file ports the GPU-free half of the /v1/audio/* endpoints: the request
// validation helpers (ref-audio base64 decoding, TTS streaming-interval
// resolution) and the byte-exact transcription response assembly. The actual
// speech-to-text, text-to-speech, and speech-to-speech computation runs on the
// STT/TTS/STS engines on the MLX backend, which are compute-gated and land with
// the rest of the compute layer.

// Upload and reference-audio size limits, ported verbatim from the reference.
const (
	// MaxAudioUploadBytes caps a transcription/process upload at 100 MB.
	MaxAudioUploadBytes = 100 * 1024 * 1024
	// MaxRefAudioBase64Bytes caps a base64 ref_audio at 20 MB (~60s of audio).
	MaxRefAudioBase64Bytes = 20 * 1024 * 1024
)

// Native TTS streaming cadence bounds, ported verbatim from the reference.
const (
	// DefaultNativeTTSStreamingIntervalSeconds is the chunk cadence used when a
	// request does not set streaming_interval.
	DefaultNativeTTSStreamingIntervalSeconds = 0.2
	// MinNativeTTSStreamingIntervalSeconds is the smallest accepted cadence.
	MinNativeTTSStreamingIntervalSeconds = 0.01
)

// AudioError carries an HTTP status and message for an audio validation
// failure. The route layer maps it onto the OpenAI error envelope. Keeping it
// HTTP-agnostic lets the api package stay free of net/http.
type AudioError struct {
	Status  int
	Message string
}

// DecodeRefAudioBase64 validates and decodes the optional base64 ref_audio from
// a text-to-speech request, mirroring the reference _decode_ref_audio_base64. A
// nil ref_audio yields no bytes and no error. When ref_audio is present, ref_text
// must be the transcript (400), the encoded size is capped (413), and the payload
// must be strict standard base64 (400).
func DecodeRefAudioBase64(refAudio *string, refText string) ([]byte, *AudioError) {
	if refAudio == nil {
		return nil, nil
	}
	if refText == "" {
		return nil, &AudioError{
			Status: 400,
			Message: "'ref_text' is required when 'ref_audio' is provided " +
				"(must be the transcript of the reference audio)",
		}
	}
	if len(*refAudio) > MaxRefAudioBase64Bytes {
		return nil, &AudioError{
			Status: 413,
			Message: fmt.Sprintf(
				"ref_audio exceeds maximum allowed size (%d bytes base64, ~60 seconds of audio)",
				MaxRefAudioBase64Bytes,
			),
		}
	}
	decoded, err := base64.StdEncoding.DecodeString(*refAudio)
	if err != nil {
		return nil, &AudioError{Status: 400, Message: "Invalid base64 encoding in 'ref_audio' field"}
	}
	return decoded, nil
}

// ResolveTTSStreamingInterval returns a native TTS streaming interval that is
// safe for the audio backend, mirroring the reference _resolve_tts_streaming_interval.
// A nil interval falls back to the default; a non-finite value or one below the
// minimum is a 400.
func ResolveTTSStreamingInterval(interval *float64) (float64, *AudioError) {
	if interval == nil {
		return DefaultNativeTTSStreamingIntervalSeconds, nil
	}
	v := *interval
	if math.IsInf(v, 0) || math.IsNaN(v) || v < MinNativeTTSStreamingIntervalSeconds {
		return 0, &AudioError{
			Status:  400,
			Message: "'streaming_interval' must be at least 0.01 seconds",
		}
	}
	return v, nil
}

// BuildTranscriptionResponse assembles the /v1/audio/transcriptions response
// body. All four fields are always present, with null where the engine did not
// supply a value, matching the AudioTranscriptionResponse pydantic model. segments
// is the per-segment list as raw JSON objects from the engine, passed through
// order-preserving. The wire form is pydantic model_dump_json (compact, non-ASCII
// passthrough), so it routes through dumpCompact, byte-for-byte with the reference.
func BuildTranscriptionResponse(text string, language *string, duration *float64, segments []json.RawMessage) string {
	languageVal := jnull()
	if language != nil {
		languageVal = jstr(*language)
	}
	durationVal := jnull()
	if duration != nil {
		durationVal = jfloat(*duration)
	}
	segmentsVal := jnull()
	if segments != nil {
		arr := jval{kind: kindArray}
		for _, raw := range segments {
			if v, ok := parseOrdered(string(raw)); ok {
				arr.arr = append(arr.arr, v)
			}
		}
		segmentsVal = arr
	}
	resp := jobj(
		"text", jstr(text),
		"language", languageVal,
		"duration", durationVal,
		"segments", segmentsVal,
	)
	return resp.dumpCompact()
}
