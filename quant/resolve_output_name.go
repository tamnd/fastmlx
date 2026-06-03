// SPDX-License-Identifier: MIT OR Apache-2.0

package quant

import (
	"regexp"
	"strconv"
)

// quantSuffixRe matches a single trailing quantization suffix, ported from
// resolve_output_name in oq.py: an oQ tag, a bit-width tag (4bit / 8_bit), an
// fp/bf dtype tag, or the mtp marker. It is anchored to the end and applied
// repeatedly so a stack of suffixes (for example -mtp-fp16-oQ4) is peeled off
// one at a time. The (?i) flag matches the reference's re.IGNORECASE; the digit
// classes are ASCII, which is all model names use.
var quantSuffixRe = regexp.MustCompile(`(?i)-(oQ[\d.]+e?|[0-9]+[_-]?bit|fp\d+|bf\d+|mtp)$`)

// ResolveOutputName builds the output model name for a quantized model, ported
// from resolve_output_name. It strips every trailing quantization suffix from
// modelName, then appends the oQ tag for the level (rendered like Python's
// f"{level:g}", so 4 becomes "4" and 3.5 stays "3.5"). A float16 dtype adds a
// -fp16 suffix (bfloat16 is the default and adds nothing), and preserveMtp adds
// a trailing -mtp so the name records that mtp tensors survived quantization.
func ResolveOutputName(modelName string, oqLevel float64, dtype string, preserveMtp bool) string {
	base := modelName
	for {
		stripped := quantSuffixRe.ReplaceAllString(base, "")
		if stripped == base {
			break
		}
		base = stripped
	}

	suffix := "-oQ" + strconv.FormatFloat(oqLevel, 'g', -1, 64)
	if dtype == "float16" {
		suffix += "-fp16"
	}
	if preserveMtp {
		suffix += "-mtp"
	}
	return base + suffix
}
