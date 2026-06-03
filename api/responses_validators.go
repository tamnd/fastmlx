// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// SerializeComplexOutput JSON-stringifies a Responses item's "output" field when
// it arrives as an array or object, porting the reference before-validator.
// Agent frameworks may send multimodal tool outputs as lists or dicts, but
// downstream code expects a string, so a complex output is flattened to its JSON
// text. The text uses ensure_ascii=True escaping with ", "/": " separators
// (Python's json.dumps default), and numbers are canonicalized the way a
// parse-then-dump round trip renders them.
//
// Only an object item with an array or object "output" is rewritten, with the
// field replaced in place so key order is preserved and the input left
// unmutated. A string, scalar, null, or absent output passes through, as does
// any non-object item.
func SerializeComplexOutput(item jval) jval {
	if item.kind != kindObject {
		return item
	}
	output, ok := item.getField("output")
	if !ok {
		return item
	}
	if output.kind == kindArray || output.kind == kindObject {
		return item.setField("output", jstr(canonicalizeJSONNumbers(output).dumpASCII()))
	}
	return item
}
