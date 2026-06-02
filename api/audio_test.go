// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type audioFixture struct {
	Transcription []struct {
		Text     string            `json:"text"`
		Language *string           `json:"language"`
		Duration *float64          `json:"duration"`
		Segments []json.RawMessage `json:"segments"`
		Expected string            `json:"expected"`
	} `json:"transcription"`
}

func loadAudioFixture(t *testing.T) audioFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/parity/audio.json")
	if err != nil {
		t.Fatal(err)
	}
	var f audioFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestBuildTranscriptionResponseParity(t *testing.T) {
	fx := loadAudioFixture(t)
	for i, c := range fx.Transcription {
		got := BuildTranscriptionResponse(c.Text, c.Language, c.Duration, c.Segments)
		if got != c.Expected {
			t.Errorf("case %d:\n got  %s\n want %s", i, got, c.Expected)
		}
	}
}

func TestDecodeRefAudioBase64(t *testing.T) {
	// Absent ref_audio: nothing decoded, no error.
	if b, e := DecodeRefAudioBase64(nil, ""); b != nil || e != nil {
		t.Errorf("nil ref_audio: got bytes=%v err=%v, want both nil", b, e)
	}

	good := "aGVsbG8=" // "hello"
	// ref_text required when ref_audio present.
	if _, e := DecodeRefAudioBase64(&good, ""); e == nil || e.Status != 400 {
		t.Errorf("missing ref_text: got %v, want 400", e)
	} else if !strings.Contains(e.Message, "ref_text") {
		t.Errorf("missing ref_text message = %q", e.Message)
	}

	// Valid decode.
	if b, e := DecodeRefAudioBase64(&good, "hello"); e != nil || string(b) != "hello" {
		t.Errorf("valid decode: got bytes=%q err=%v", b, e)
	}

	// Invalid base64.
	bad := "not valid base64!!!"
	if _, e := DecodeRefAudioBase64(&bad, "x"); e == nil || e.Status != 400 {
		t.Errorf("bad base64: got %v, want 400", e)
	} else if !strings.Contains(e.Message, "Invalid base64") {
		t.Errorf("bad base64 message = %q", e.Message)
	}

	// Oversize payload (413). A string longer than the cap; content does not
	// need to be decoded because the size check runs first.
	big := strings.Repeat("A", MaxRefAudioBase64Bytes+4)
	if _, e := DecodeRefAudioBase64(&big, "x"); e == nil || e.Status != 413 {
		t.Errorf("oversize: got %v, want 413", e)
	}
}

func TestResolveTTSStreamingInterval(t *testing.T) {
	if v, e := ResolveTTSStreamingInterval(nil); e != nil || v != DefaultNativeTTSStreamingIntervalSeconds {
		t.Errorf("nil interval: got %v err=%v, want default", v, e)
	}

	ok := 0.5
	if v, e := ResolveTTSStreamingInterval(&ok); e != nil || v != 0.5 {
		t.Errorf("valid interval: got %v err=%v", v, e)
	}

	atMin := MinNativeTTSStreamingIntervalSeconds
	if v, e := ResolveTTSStreamingInterval(&atMin); e != nil || v != atMin {
		t.Errorf("at-minimum interval: got %v err=%v, want accepted", v, e)
	}

	tooSmall := 0.001
	if _, e := ResolveTTSStreamingInterval(&tooSmall); e == nil || e.Status != 400 {
		t.Errorf("below minimum: got %v, want 400", e)
	}
}

func BenchmarkBuildTranscriptionResponse(b *testing.B) {
	lang := "en"
	dur := 12.5
	segments := []json.RawMessage{
		json.RawMessage(`{"id":0,"start":0.0,"end":1.5,"text":"hello"}`),
		json.RawMessage(`{"id":1,"start":1.5,"end":3.0,"text":"world"}`),
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildTranscriptionResponse("hello world", &lang, &dur, segments)
	}
}
