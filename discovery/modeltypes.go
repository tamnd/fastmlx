// SPDX-License-Identifier: MIT OR Apache-2.0

// Package discovery classifies models on disk into engine-routable types
// for engine routing. The type and architecture sets are kept
// stable; modeltypes_test.go guards against drift.
package discovery

// ModelType is the discriminator for engine routing.
type ModelType string

const (
	TypeLLM       ModelType = "llm"
	TypeVLM       ModelType = "vlm"
	TypeEmbedding ModelType = "embedding"
	TypeReranker  ModelType = "reranker"
	TypeAudioSTT  ModelType = "audio_stt"
	TypeAudioTTS  ModelType = "audio_tts"
	TypeAudioSTS  ModelType = "audio_sts"
)

// EngineType is the engine implementation a model routes to.
type EngineType string

const (
	EngineBatched   EngineType = "batched"
	EngineVLM       EngineType = "vlm"
	EngineEmbedding EngineType = "embedding"
	EngineReranker  EngineType = "reranker"
	EngineAudioSTT  EngineType = "audio_stt"
	EngineAudioTTS  EngineType = "audio_tts"
	EngineAudioSTS  EngineType = "audio_sts"
)

// EngineFor maps a ModelType to its default EngineType.
func EngineFor(t ModelType) EngineType {
	switch t {
	case TypeVLM:
		return EngineVLM
	case TypeEmbedding:
		return EngineEmbedding
	case TypeReranker:
		return EngineReranker
	case TypeAudioSTT:
		return EngineAudioSTT
	case TypeAudioTTS:
		return EngineAudioTTS
	case TypeAudioSTS:
		return EngineAudioSTS
	default:
		return EngineBatched
	}
}

// VLMModelTypes lists the vision-language model types.
var VLMModelTypes = set(
	"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5_moe",
	"gemma3", "gemma4", "llava", "llava_next", "llava-qwen2", "llava_qwen2",
	"mllama", "idefics3", "internvl_chat", "phi3_v", "paligemma", "mistral3",
	"pixtral", "molmo", "molmo2", "bunny_llama", "multi_modality", "florence2",
	"deepseekocr", "deepseekocr_2", "dots_ocr", "glm_ocr", "minicpmv",
	"phi4_siglip", "phi4mm", "youtu_vl",
)

// VLMArchitectures lists the vision-language model architectures.
var VLMArchitectures = set(
	"LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration",
	"Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration",
	"MllamaForConditionalGeneration", "Gemma3ForConditionalGeneration",
	"Gemma4ForConditionalGeneration", "InternVLChatModel",
	"Idefics3ForConditionalGeneration", "PaliGemmaForConditionalGeneration",
	"Phi3VForCausalLM", "Pixtral", "MolmoForCausalLM",
	"Molmo2ForConditionalGeneration", "LlavaQwen2ForCausalLM",
	"Florence2ForConditionalGeneration",
)

// EmbeddingModelTypes lists the embedding model types.
var EmbeddingModelTypes = set(
	"bert", "xlm-roberta", "xlm_roberta", "modernbert", "siglip",
	"colqwen2_5", "colqwen2-5", "lfm2",
)

// AmbiguousEmbeddingModelTypes lists the ambiguous embedding model types:
// types that have both embedding and LLM variants and need architecture
// disambiguation.
var AmbiguousEmbeddingModelTypes = set("qwen3", "gemma3-text", "gemma3_text")

// EmbeddingArchitectures lists the embedding architectures.
var EmbeddingArchitectures = set(
	"BertModel", "BertForMaskedLM", "XLMRobertaModel", "XLMRobertaForMaskedLM",
	"ModernBertModel", "ModernBertForMaskedLM", "Qwen3ForTextEmbedding",
	"SiglipModel", "SiglipVisionModel", "SiglipTextModel",
)

// SupportedRerankerArchitectures lists the supported reranker architectures.
var SupportedRerankerArchitectures = set(
	"ModernBertForSequenceClassification", "XLMRobertaForSequenceClassification",
	"JinaForRanking",
)

// UnsupportedRerankerArchitectures lists reranker architectures that are not supported.
var UnsupportedRerankerArchitectures = set(
	"BertForSequenceClassification", "Qwen3ForSequenceClassification",
)

// CausalLMRerankerArchitectures lists causal-LM architectures used for reranking.
var CausalLMRerankerArchitectures = set("Qwen3ForCausalLM")

// CausalLMEmbeddingArchitectures lists causal-LM architectures used for embeddings.
var CausalLMEmbeddingArchitectures = set("Qwen3ForCausalLM")

// MultimodalRerankerArchitectures lists multimodal reranker architectures.
var MultimodalRerankerArchitectures = set("Qwen3VLForConditionalGeneration")

// MultimodalEmbeddingArchitectures lists multimodal embedding architectures.
var MultimodalEmbeddingArchitectures = set("Qwen3VLForConditionalGeneration")

// RerankerArchitectures is the union of supported and unsupported reranker
// architectures used for type detection.
var RerankerArchitectures = union(SupportedRerankerArchitectures, UnsupportedRerankerArchitectures)

// UnsupportedArchitectures and UnsupportedModelTypes name model families that
// discovery skips entirely. Both are empty in the reference today (whisper and
// qwen3_tts moved to the audio types), but the detector and these sets are
// ported so a future entry takes effect without code changes.
var UnsupportedArchitectures = set()

// UnsupportedModelTypes is the model_type skip set (empty upstream today).
var UnsupportedModelTypes = set()

func set(items ...string) map[string]struct{} {
	m := make(map[string]struct{}, len(items))
	for _, it := range items {
		m[it] = struct{}{}
	}
	return m
}

func union(maps ...map[string]struct{}) map[string]struct{} {
	out := map[string]struct{}{}
	for _, m := range maps {
		for k := range m {
			out[k] = struct{}{}
		}
	}
	return out
}

// has reports whether key is in m.
func has(m map[string]struct{}, key string) bool {
	_, ok := m[key]
	return ok
}
