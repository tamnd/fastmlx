// SPDX-License-Identifier: MIT OR Apache-2.0

// Package imageutil holds the pure, decode-free image helpers used by the prefix
// cache: hashing a request's images so identical image inputs share a cached
// prompt prefix. The decode from a URL or base64 payload to pixels is the
// caller's seam; these functions take an already-decoded image.
package imageutil

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
)

// Image is a decoded image reduced to what the hash depends on: its pixel
// dimensions and its raw RGB bytes (three bytes per pixel, row major). The
// reference derives this from PIL's img.size and img.convert("RGB").tobytes();
// performing that decode is the caller's responsibility.
type Image struct {
	Width  int
	Height int
	RGB    []byte
}

// ComputeImageHash hashes a list of images into one hex SHA256 digest for prefix
// cache deduplication, mixing in each image's dimensions before its pixel bytes
// so differently sized images never collide. An empty list has no hash, so it
// returns the empty string (the reference's None).
func ComputeImageHash(images []Image) string {
	if len(images) == 0 {
		return ""
	}
	hasher := sha256.New()
	for _, img := range images {
		hasher.Write([]byte(dims(img)))
		hasher.Write(img.RGB)
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

// ComputePerImageHashes hashes each image independently, returning one hex
// SHA256 digest per image in order.
func ComputePerImageHashes(images []Image) []string {
	hashes := make([]string, 0, len(images))
	for _, img := range images {
		hasher := sha256.New()
		hasher.Write([]byte(dims(img)))
		hasher.Write(img.RGB)
		hashes = append(hashes, hex.EncodeToString(hasher.Sum(nil)))
	}
	return hashes
}

// dims renders an image's size as "WxH", matching the reference's
// f"{img.size[0]}x{img.size[1]}".
func dims(img Image) string {
	return strconv.Itoa(img.Width) + "x" + strconv.Itoa(img.Height)
}
