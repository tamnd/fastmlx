// SPDX-License-Identifier: MIT OR Apache-2.0

// Package audioutil holds the byte-level WAV helpers shared by the audio
// engines (speech-to-text, text-to-speech, speech-to-speech). They are the
// GPU-free codec seam between the model, which produces or consumes raw float
// samples, and the HTTP layer, which speaks WAV. Producing the float samples
// (running the model) stays upstream; everything here is pure byte work.
package audioutil

import (
	"encoding/binary"
	"errors"
)

// DefaultSampleRate is the sample rate assumed when a model does not report one.
const DefaultSampleRate = 24000

// maxWAVChunkSize is the sentinel size a streaming WAV header carries when the
// total length is not yet known.
const maxWAVChunkSize = 0xFFFFFFFF

// WAVHeader builds a 44-byte PCM WAV header for streamed or unknown-length
// audio: both the RIFF and data chunk sizes are left at the 0xFFFFFFFF sentinel
// since the final length is not known when streaming begins.
func WAVHeader(sampleRate, channels, sampleWidth int) []byte {
	blockAlign := channels * sampleWidth
	byteRate := sampleRate * blockAlign
	bitsPerSample := sampleWidth * 8

	h := make([]byte, 0, 44)
	h = append(h, "RIFF"...)
	h = appendU32(h, maxWAVChunkSize)
	h = append(h, "WAVE"...)
	h = append(h, "fmt "...)
	h = appendU32(h, 16)
	h = appendU16(h, 1) // PCM
	h = appendU16(h, uint16(channels))
	h = appendU32(h, uint32(sampleRate))
	h = appendU32(h, uint32(byteRate))
	h = appendU16(h, uint16(blockAlign))
	h = appendU16(h, uint16(bitsPerSample))
	h = append(h, "data"...)
	h = appendU32(h, maxWAVChunkSize)
	return h
}

// AudioFloat32ToWAVBytes encodes float32 samples in [-1, 1] as a finite mono
// 16-bit PCM WAV. Each sample is clipped to [-1, 1] and scaled by 32767 with the
// arithmetic done in float32 and truncated toward zero, matching the reference's
// numpy float32 path exactly. The RIFF and data chunk sizes carry the real
// lengths (unlike the streaming header).
func AudioFloat32ToWAVBytes(samples []float32, sampleRate int) []byte {
	const (
		channels    = 1
		sampleWidth = 2
	)
	pcm := make([]byte, 0, len(samples)*2)
	for _, s := range samples {
		if s > 1 {
			s = 1
		} else if s < -1 {
			s = -1
		}
		v := int16(s * 32767)
		pcm = appendU16(pcm, uint16(v))
	}

	blockAlign := channels * sampleWidth
	byteRate := sampleRate * blockAlign
	dataLen := len(pcm)

	out := make([]byte, 0, 44+dataLen)
	out = append(out, "RIFF"...)
	out = appendU32(out, uint32(36+dataLen))
	out = append(out, "WAVE"...)
	out = append(out, "fmt "...)
	out = appendU32(out, 16)
	out = appendU16(out, 1) // PCM
	out = appendU16(out, channels)
	out = appendU32(out, uint32(sampleRate))
	out = appendU32(out, uint32(byteRate))
	out = appendU16(out, uint16(blockAlign))
	out = appendU16(out, sampleWidth*8)
	out = append(out, "data"...)
	out = appendU32(out, uint32(dataLen))
	out = append(out, pcm...)
	return out
}

// WAVBytesToPCMFrames parses a PCM WAV and returns its sample rate, channel
// count, sample width in bytes, and the raw PCM frame bytes. Like the reference
// readframes, it returns whole frames only, so a data chunk is truncated to a
// multiple of the frame size.
func WAVBytesToPCMFrames(wavBytes []byte) (sampleRate, channels, sampleWidth int, pcm []byte, err error) {
	if len(wavBytes) < 12 || string(wavBytes[0:4]) != "RIFF" || string(wavBytes[8:12]) != "WAVE" {
		return 0, 0, 0, nil, errors.New("not a RIFF/WAVE file")
	}

	var haveFmt, haveData bool
	var bitsPerSample int
	var dataStart, dataLen int

	off := 12
	for off+8 <= len(wavBytes) {
		id := string(wavBytes[off : off+4])
		size := int(binary.LittleEndian.Uint32(wavBytes[off+4 : off+8]))
		body := off + 8
		if body+size > len(wavBytes) {
			size = len(wavBytes) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return 0, 0, 0, nil, errors.New("short fmt chunk")
			}
			channels = int(binary.LittleEndian.Uint16(wavBytes[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(wavBytes[body+4 : body+8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(wavBytes[body+14 : body+16]))
			haveFmt = true
		case "data":
			dataStart = body
			dataLen = size
			haveData = true
		}
		// Chunks are word-aligned: an odd size is followed by a pad byte.
		off = body + size
		if size%2 == 1 {
			off++
		}
	}

	if !haveFmt || !haveData {
		return 0, 0, 0, nil, errors.New("missing fmt or data chunk")
	}
	sampleWidth = bitsPerSample / 8
	frameSize := channels * sampleWidth
	if frameSize <= 0 {
		return 0, 0, 0, nil, errors.New("invalid frame size")
	}
	nframes := dataLen / frameSize
	pcm = wavBytes[dataStart : dataStart+nframes*frameSize]
	return sampleRate, channels, sampleWidth, pcm, nil
}

func appendU16(b []byte, v uint16) []byte {
	return append(b, byte(v), byte(v>>8))
}

func appendU32(b []byte, v uint32) []byte {
	return append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}
