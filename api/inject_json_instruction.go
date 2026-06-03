// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// InjectJSONInstruction folds a JSON/format instruction into a chat message
// list, porting _inject_json_instruction. It locates the first system message
// and appends the instruction after a blank line; with no system message it
// prepends a fresh one. The reference's pydantic-object branch is the one seam
// (the API path passes dict messages), so only the dict branch is ported.
//
// The append reproduces the reference f-string rendering of the existing
// content: a missing content key contributes "" and a present null content
// renders as the Python "None", both followed by "\n\n" and the instruction. The
// returned slice is a fresh copy; the system message keeps its key order with
// content replaced in place.
func InjectJSONInstruction(messages []jval, instruction string) []jval {
	out := make([]jval, len(messages))
	copy(out, messages)

	systemIdx := -1
	for i, msg := range out {
		if msg.getString("role") == "system" {
			systemIdx = i
			break
		}
	}

	if systemIdx >= 0 {
		msg := out[systemIdx]
		existing := ""
		if c, ok := msg.getField("content"); ok {
			existing = pythonStrValue(c)
		}
		out[systemIdx] = msg.setField("content", jstr(existing+"\n\n"+instruction))
		return out
	}

	sys := jobj("role", jstr("system"), "content", jstr(instruction))
	return append([]jval{sys}, out...)
}

// ShouldStoreResponse ports _should_store_response: the OpenAI Responses API
// stores a response unless the store flag is explicitly false, so only a literal
// false disables storage (a missing/null flag or true both enable it).
func ShouldStoreResponse(storeFlag jval) bool {
	return !(storeFlag.kind == kindBool && storeFlag.s == "false")
}
