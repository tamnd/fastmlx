// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// JSON-output parsing for response_format, ported from parse_json_output in
// tool_calling.py. It decides whether a finished generation needs JSON
// extraction, runs the shared ExtractJSONFromText strategies, and (for
// json_schema) defers the schema verdict to an injected validator.

// JSONSchemaValidator is the schema-validation seam parse_json_output needs.
// The reference wraps the jsonschema library, whose error-message strings are
// library- and version-specific, so it is injected here rather than reproduced
// (the same boundary documented in structured_output.go). Validate reports
// whether data matches schema and, when it does not, the bare error message that
// ParseJSONOutput prefixes with "JSON Schema validation failed: ".
type JSONSchemaValidator interface {
	Validate(data, schema jval) (bool, string)
}

// JSONParseResult is the four-value return of parse_json_output. Parsed is nil
// when no JSON was applicable or extraction failed (the reference's None), and
// ErrorMessage is "" when there is no error (the reference's None).
type JSONParseResult struct {
	CleanedText  string
	Parsed       *jval
	Valid        bool
	ErrorMessage string
}

// ParseJSONOutput parses JSON from model output when a response_format is set,
// ported from parse_json_output. A nil response_format or the "text" type
// returns the text untouched with no parsed value. For "json_object" and
// "json_schema" it extracts JSON via ExtractJSONFromText; extraction failure is
// an invalid result carrying the fixed "Failed to extract valid JSON from
// output" message. "json_object" accepts any extracted JSON. "json_schema"
// validates the extracted value against the schema only when the schema is
// truthy (a non-empty object, matching Python's `if schema:`), wrapping a
// validator failure as "JSON Schema validation failed: <message>". An
// unrecognized type that nonetheless yields extractable JSON is treated as text:
// it reports valid with no parsed value, matching the reference's final return.
//
// The schema is read from the response_format's json_schema sub-object (the
// "schema" field), exactly as BuildJSONSystemPrompt reads it.
func ParseJSONOutput(text string, rf *ResponseFormat, validator JSONSchemaValidator) JSONParseResult {
	if rf == nil {
		return JSONParseResult{CleanedText: text, Valid: true}
	}

	formatType := rf.Type
	if formatType == "" {
		formatType = "text"
	}
	if formatType == "text" {
		return JSONParseResult{CleanedText: text, Valid: true}
	}

	parsed, ok := ExtractJSONFromText(text)
	if !ok {
		return JSONParseResult{
			CleanedText:  text,
			Valid:        false,
			ErrorMessage: "Failed to extract valid JSON from output",
		}
	}

	switch formatType {
	case "json_object":
		return JSONParseResult{CleanedText: text, Parsed: &parsed, Valid: true}

	case "json_schema":
		spec, _ := parseOrdered(string(rf.JSONSchema))
		schema := spec.getOr("schema", jval{kind: kindObject})
		if pythonTruthy(schema) {
			valid, errMsg := validator.Validate(parsed, schema)
			if !valid {
				return JSONParseResult{
					CleanedText:  text,
					Parsed:       &parsed,
					Valid:        false,
					ErrorMessage: "JSON Schema validation failed: " + errMsg,
				}
			}
		}
		return JSONParseResult{CleanedText: text, Parsed: &parsed, Valid: true}

	default: // unknown type with extractable JSON: treat as text
		return JSONParseResult{CleanedText: text, Valid: true}
	}
}
