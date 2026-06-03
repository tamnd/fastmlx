// SPDX-License-Identifier: MIT OR Apache-2.0

package api

import (
	"fmt"
	"strings"
)

// Listwise prompt construction for the Jina-style CausalLM reranker. The model
// that consumes this prompt and reads its logits lives behind the GPU seam; the
// prompt text itself is pure string assembly and is built here.

// jinaRerankSystemPrompt is the fixed system instruction for the listwise
// reranker, reproduced verbatim from the reference.
const jinaRerankSystemPrompt = "You are a search relevance expert who can determine a ranking of the " +
	"passages based on how relevant they are to the query. If the query is " +
	"a question, how relevant a passage is depends on how well it answers " +
	"the question. If not, try to analyze the intent of the query and " +
	"assess how well each passage satisfies the intent. If an instruction " +
	"is provided, you should follow the instruction when determining the " +
	"ranking."

// jinaTextReplacer strips the reranker's own special tokens out of user text so
// they cannot break the prompt framing. The replacements are independent (none
// is a substring of another), so a single pass matches the reference's
// sequential str.replace calls.
var jinaTextReplacer = strings.NewReplacer(
	"<|embed_token|>", " ",
	"<|rerank_token|>", " ",
	"<|score_token|>", " ",
	"<|im_start|>", " ",
	"<|im_end|>", " ",
)

// sanitizeJinaText removes conflicting special tokens from user-provided text
// and trims surrounding whitespace.
func sanitizeJinaText(text string) string {
	return strings.TrimSpace(jinaTextReplacer.Replace(text))
}

// FormatJinaRerankPrompt builds the listwise Jina reranking prompt for a query
// and its candidate documents. Query, documents, and instruction are each
// sanitized first; the instruction block is included only when it survives
// sanitization as non-empty (so an absent, empty, or whitespace-only
// instruction is omitted). Passages are numbered from zero in input order.
func FormatJinaRerankPrompt(query string, documents []string, instruction string) string {
	sanitizedQuery := sanitizeJinaText(query)
	sanitizedDocs := make([]string, len(documents))
	for i, doc := range documents {
		sanitizedDocs[i] = sanitizeJinaText(doc)
	}

	userContent := fmt.Sprintf(
		"I will provide you with %d passages, each indicated by a numerical "+
			"identifier. Rank the passages based on their relevance to query: %s\n",
		len(sanitizedDocs), sanitizedQuery)

	if si := sanitizeJinaText(instruction); si != "" {
		userContent += "<instruct>\n" + si + "\n</instruct>\n"
	}

	docPrompts := make([]string, len(sanitizedDocs))
	for i, doc := range sanitizedDocs {
		docPrompts[i] = fmt.Sprintf("<passage id=\"%d\">\n%s<|embed_token|>\n</passage>", i, doc)
	}
	userContent += strings.Join(docPrompts, "\n") + "\n"
	userContent += "<query>\n" + sanitizedQuery + "<|rerank_token|>\n</query>"

	return "<|im_start|>system\n" +
		jinaRerankSystemPrompt +
		"<|im_end|>\n" +
		"<|im_start|>user\n" +
		userContent +
		"<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<think>\n\n</think>\n\n"
}
