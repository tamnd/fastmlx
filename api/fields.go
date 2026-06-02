// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import "encoding/json"

// StopField decodes a JSON value that may be a single string or an array of
// strings (OpenAI stop / prompt / input fields). The decoded form is always a
// slice.
type StopField []string

// UnmarshalJSON accepts a string, an array of strings, or null.
func (s *StopField) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		*s = nil
		return nil
	}
	if data[0] == '[' {
		var arr []string
		if err := json.Unmarshal(data, &arr); err != nil {
			return err
		}
		*s = arr
		return nil
	}
	var one string
	if err := json.Unmarshal(data, &one); err != nil {
		return err
	}
	*s = []string{one}
	return nil
}

// MarshalJSON emits a bare string for a single element and an array otherwise,
// matching how OpenAI echoes these fields.
func (s StopField) MarshalJSON() ([]byte, error) {
	if len(s) == 1 {
		return json.Marshal(s[0])
	}
	return json.Marshal([]string(s))
}

// First returns the first element or "" when empty (handy for single-prompt use).
func (s StopField) First() string {
	if len(s) == 0 {
		return ""
	}
	return s[0]
}

// Error is the OpenAI error envelope body.
type Error struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

// ErrorEnvelope wraps Error under the "error" key, as OpenAI returns it.
type ErrorEnvelope struct {
	Error Error `json:"error"`
}

// NewError builds an error envelope.
func NewError(message, typ, code string) ErrorEnvelope {
	return ErrorEnvelope{Error: Error{Message: message, Type: typ, Code: code}}
}
