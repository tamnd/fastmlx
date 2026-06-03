// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// StatusToErrorType maps an HTTP status code to the OpenAI error "type" string,
// porting _status_to_error_type: 401 is an authentication error, 404 a
// not-found error, 429 a rate-limit error, any 5xx a server error, and anything
// else an invalid-request error.
func StatusToErrorType(statusCode int) string {
	switch {
	case statusCode == 401:
		return "authentication_error"
	case statusCode == 404:
		return "not_found_error"
	case statusCode == 429:
		return "rate_limit_error"
	case statusCode >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

// OpenAIErrorBody builds the OpenAI-compatible error response body, porting
// _openai_error_body. The message, param, and code values pass through verbatim
// (param and code are usually null) and the error type is derived from the
// status code. The fixed key order is message, type, param, code.
func OpenAIErrorBody(message jval, statusCode int, param, code jval) jval {
	return jobj("error", jobj(
		"message", message,
		"type", jstr(StatusToErrorType(statusCode)),
		"param", param,
		"code", code,
	))
}
