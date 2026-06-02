// SPDX-License-Identifier: MIT OR Apache-2.0

package discovery

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, dir, config string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDetectModelType(t *testing.T) {
	cases := []struct {
		name   string
		dir    string
		config string
		want   ModelType
	}{
		{"qwen3 llm", "Qwen3-4B", `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`, TypeLLM},
		{"vlm arch", "x", `{"architectures":["Qwen2VLForConditionalGeneration"]}`, TypeVLM},
		{"vlm type", "x", `{"model_type":"llava"}`, TypeVLM},
		{"embedding arch", "x", `{"architectures":["BertModel"]}`, TypeEmbedding},
		{"embedding type", "x", `{"model_type":"modernbert"}`, TypeEmbedding},
		{"reranker arch", "x", `{"architectures":["XLMRobertaForSequenceClassification"]}`, TypeReranker},
		{"causal reranker by dirname", "Qwen3-Reranker-4B", `{"architectures":["Qwen3ForCausalLM"]}`, TypeReranker},
		{"vision subconfig heuristic", "x", `{"model_type":"unknownthing","vision_config":{"a":1}}`, TypeVLM},
		{"plain llm fallback", "x", `{"model_type":"mistral"}`, TypeLLM},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), c.dir)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			writeConfig(t, dir, c.config)
			got, err := DetectModelType(dir)
			if err != nil {
				t.Fatalf("DetectModelType: %v", err)
			}
			if got != c.want {
				t.Errorf("DetectModelType = %q, want %q", got, c.want)
			}
		})
	}
}

func TestReadModelContextLength(t *testing.T) {
	cases := []struct {
		config string
		want   int
	}{
		{`{"max_position_embeddings":32768}`, 32768},
		{`{"seq_length":4096}`, 4096},
		{`{"text_config":{"max_position_embeddings":8192}}`, 8192},
		{`{"foo":1}`, 0},
		{`{"max_position_embeddings":1e30}`, 0}, // non-integral float rejected
	}
	for _, c := range cases {
		dir := t.TempDir()
		writeConfig(t, dir, c.config)
		if got := ReadModelContextLength(dir); got != c.want {
			t.Errorf("ReadModelContextLength(%s) = %d, want %d", c.config, got, c.want)
		}
	}
}

func TestEngineFor(t *testing.T) {
	if EngineFor(TypeLLM) != EngineBatched {
		t.Error("llm should route to batched")
	}
	if EngineFor(TypeAudioTTS) != EngineAudioTTS {
		t.Error("audio_tts routing")
	}
}
