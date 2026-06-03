// SPDX-License-Identifier: MIT OR Apache-2.0

package audioutil

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type wavFixture struct {
	Header []struct {
		SampleRate  int    `json:"sample_rate"`
		Channels    int    `json:"channels"`
		SampleWidth int    `json:"sample_width"`
		Hex         string `json:"hex"`
	} `json:"header"`
	Encode []struct {
		Samples    []float64 `json:"samples"`
		SampleRate int       `json:"sample_rate"`
		Hex        string    `json:"hex"`
	} `json:"encode"`
	Parse []struct {
		WavHex      string `json:"wav_hex"`
		SampleRate  int    `json:"sample_rate"`
		Channels    int    `json:"channels"`
		SampleWidth int    `json:"sample_width"`
		PCMHex      string `json:"pcm_hex"`
	} `json:"parse"`
}

func loadWAVFixture(t *testing.T) wavFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/wav.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx wavFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return fx
}

func TestWAVHeader(t *testing.T) {
	fx := loadWAVFixture(t)
	for i, c := range fx.Header {
		got := hex.EncodeToString(WAVHeader(c.SampleRate, c.Channels, c.SampleWidth))
		if got != c.Hex {
			t.Errorf("header[%d] sr=%d ch=%d w=%d:\n got %s\nwant %s", i, c.SampleRate, c.Channels, c.SampleWidth, got, c.Hex)
		}
	}
}

func TestAudioFloat32ToWAVBytes(t *testing.T) {
	fx := loadWAVFixture(t)
	for i, c := range fx.Encode {
		samples := make([]float32, len(c.Samples))
		for j, s := range c.Samples {
			samples[j] = float32(s)
		}
		got := hex.EncodeToString(AudioFloat32ToWAVBytes(samples, c.SampleRate))
		if got != c.Hex {
			t.Errorf("encode[%d] sr=%d n=%d:\n got %s\nwant %s", i, c.SampleRate, len(samples), got, c.Hex)
		}
	}
}

func TestWAVBytesToPCMFrames(t *testing.T) {
	fx := loadWAVFixture(t)
	for i, c := range fx.Parse {
		wav, err := hex.DecodeString(c.WavHex)
		if err != nil {
			t.Fatalf("parse[%d] bad wav_hex: %v", i, err)
		}
		sr, ch, sw, pcm, err := WAVBytesToPCMFrames(wav)
		if err != nil {
			t.Fatalf("parse[%d]: %v", i, err)
		}
		if sr != c.SampleRate || ch != c.Channels || sw != c.SampleWidth {
			t.Errorf("parse[%d] meta: got sr=%d ch=%d w=%d, want sr=%d ch=%d w=%d", i, sr, ch, sw, c.SampleRate, c.Channels, c.SampleWidth)
		}
		if got := hex.EncodeToString(pcm); got != c.PCMHex {
			t.Errorf("parse[%d] pcm:\n got %s\nwant %s", i, got, c.PCMHex)
		}
	}
}

func TestWAVBytesToPCMFramesRejectsNonRIFF(t *testing.T) {
	cases := [][]byte{
		nil,
		[]byte("RIFF"),
		[]byte("MTHd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0"),
		append([]byte("RIFF\x24\x00\x00\x00OGGS"), make([]byte, 8)...),
	}
	for i, c := range cases {
		if _, _, _, _, err := WAVBytesToPCMFrames(c); err == nil {
			t.Errorf("case %d: expected error for non-WAVE input", i)
		}
	}
}

func TestAudioFloat32RoundTrip(t *testing.T) {
	// Encoding then parsing should recover the same PCM the encoder wrote.
	samples := []float32{0.5, -0.5, 0.25, -0.25, 0.0, 1.0, -1.0}
	wav := AudioFloat32ToWAVBytes(samples, DefaultSampleRate)
	sr, ch, sw, pcm, err := WAVBytesToPCMFrames(wav)
	if err != nil {
		t.Fatalf("round trip parse: %v", err)
	}
	if sr != DefaultSampleRate || ch != 1 || sw != 2 {
		t.Errorf("round trip meta: sr=%d ch=%d w=%d", sr, ch, sw)
	}
	if len(pcm) != len(samples)*2 {
		t.Errorf("round trip pcm length: got %d, want %d", len(pcm), len(samples)*2)
	}
}

func BenchmarkAudioFloat32ToWAVBytes(b *testing.B) {
	samples := make([]float32, 24000)
	for i := range samples {
		samples[i] = float32(i%2000)/1000 - 1
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = AudioFloat32ToWAVBytes(samples, DefaultSampleRate)
	}
}

func BenchmarkWAVBytesToPCMFrames(b *testing.B) {
	samples := make([]float32, 24000)
	for i := range samples {
		samples[i] = float32(i%2000)/1000 - 1
	}
	wav := AudioFloat32ToWAVBytes(samples, DefaultSampleRate)
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _ = WAVBytesToPCMFrames(wav)
	}
}
