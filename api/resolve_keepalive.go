// SPDX-License-Identifier: MIT OR Apache-2.0

package api

// SSE keepalive frames, ported verbatim from the reference module constants.
// The chunk frames carry a sentinel id and a "keepalive" model so they parse as
// valid no-op stream events for clients that cannot read SSE comment lines; the
// comment frame is the legacy keepalive and the ping is the Anthropic event.
const (
	keepaliveComment         = ": keep-alive\n\n"
	keepaliveChatChunk       = "data: {\"id\":\"chatcmpl-keepalive\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"keepalive\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n"
	keepaliveCompletionChunk = "data: {\"id\":\"cmpl-keepalive\",\"object\":\"text_completion\",\"created\":0,\"model\":\"keepalive\",\"choices\":[{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":null}]}\n\n"
	keepaliveAnthropicPing   = "event: ping\ndata: {\"type\":\"ping\"}\n\n"
)

// ResolveKeepalive picks the wire-level keepalive frame for an API protocol,
// porting _resolve_keepalive from server.py. The reference reads the configured
// keepalive mode from global settings (defaulting to "chunk" when no global
// settings are present); that read is server-state-bound, so mode is injected
// here as the already-resolved value and only the pure decision is ported.
//
// The second return reports whether a frame applies: it is false (with an empty
// string) when the mode disables keepalive ("off") or when the protocol has no
// frame ("openai_responses", or anything unrecognized, which is the reference's
// final fall-through to None). The "comment" mode returns the legacy SSE comment for
// every protocol; otherwise the chunk frame is chosen per protocol. An
// unrecognized mode falls through to the per-protocol chunk frames, matching the
// reference, since only "off" and "comment" are special-cased before the switch.
func ResolveKeepalive(mode, protocol string) (string, bool) {
	if mode == "off" {
		return "", false
	}
	if mode == "comment" {
		return keepaliveComment, true
	}
	switch protocol {
	case "openai_chat":
		return keepaliveChatChunk, true
	case "openai_completion":
		return keepaliveCompletionChunk, true
	case "anthropic":
		return keepaliveAnthropicPing, true
	default:
		return "", false
	}
}

// ChatKeepaliveChunk builds a chat keepalive frame carrying the stream's own
// completion id instead of the static sentinel, porting _chat_keepalive_chunk.
// Strict OpenAI stream accumulators latch the first chunk's id and silently drop
// later chunks whose id differs; emitting the keepalive under the stream's own
// response id makes it a true no-op for those clients while still parsing as a
// data event for clients that cannot read SSE comment lines. The "keepalive"
// model marker is the literal protocol sentinel and is kept verbatim.
func ChatKeepaliveChunk(responseID string) string {
	return "data: {\"id\":\"" + responseID + "\",\"object\":\"chat.completion.chunk\"," +
		"\"created\":0,\"model\":\"keepalive\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n"
}
