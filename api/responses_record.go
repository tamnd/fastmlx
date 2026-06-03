// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// NormalizeResponseRecord ports the response store's _normalize_record: it turns
// either an already-persisted state record or a bare public response into the
// canonical record shape used in memory, so a reload from disk is idempotent.
//
// The response id is resolved by the caller (from the record's own response_id
// or its public_response.id) and passed in, keeping the disk and id bookkeeping
// out of this pure transform. The only dependency, recomputing output_messages
// from the response output, is the already-ported NormalizeResponseOutputToMessages.
//
// When the input already carries a public_response key it is a saved record:
// deep-copy it and fill in any missing top-level fields without disturbing the
// ones already present (setdefault), so existing output_messages or created_at
// survive a reload untouched. Otherwise the input is a fresh public response:
// deep-copy it, default its id to the resolved id, and build the record around
// it with input_messages empty and output_messages derived from its output.
func NormalizeResponseRecord(responseID string, responseData jval) jval {
	if responseData.hasField("public_response") {
		record := cloneJval(responseData)
		pub := record.getOr("public_response", jval{kind: kindObject})
		record = record.setDefault("response_id", jstr(responseID))
		record = record.setDefault("created_at", pub.getOr("created_at", jint(0)))
		record = record.setDefault("previous_response_id", pub.getOr("previous_response_id", jnull()))
		record = record.setDefault("input_messages", jval{kind: kindArray})
		if !record.hasField("output_messages") {
			output, _ := pub.getField("output")
			record = record.setField("output_messages", jarrOf(NormalizeResponseOutputToMessages(output)))
		}
		return record
	}

	pub := cloneJval(responseData)
	pub = pub.setDefault("id", jstr(responseID))
	output, _ := pub.getField("output")
	return jobj(
		"response_id", jstr(responseID),
		"previous_response_id", cloneJval(pub.getOr("previous_response_id", jnull())),
		"input_messages", jval{kind: kindArray},
		"output_messages", jarrOf(NormalizeResponseOutputToMessages(output)),
		"public_response", pub,
		"created_at", cloneJval(pub.getOr("created_at", jint(0))),
	)
}

// setDefault appends key=val only when key is absent, mirroring dict.setdefault:
// a present key keeps its value (and the object's existing field order).
func (v jval) setDefault(key string, val jval) jval {
	if v.hasField(key) {
		return v
	}
	return v.setField(key, val)
}
