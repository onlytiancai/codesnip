package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// The local Ollama server and the translation model it runs. The
// environment overrides them, so the app keeps working when Ollama lives
// somewhere else or runs another model.
const (
	defaultOllamaHost  = "http://localhost:11434"
	defaultOllamaModel = "s2021008840/hy-mt2:1.8b"
	ollamaHostEnv      = "OLLAMA_HOST"
	ollamaModelEnv     = "OLLAMA_MODEL"
	translateTimeout   = 60 * time.Second
	maxErrorBodyLength = 200
)

// lang is one language the model translates between. The label is what the
// interface shows; the name is what the model is asked for, in English,
// because that is the only form of the instruction it follows.
type lang struct {
	label string
	name  string
}

// langs are the languages offered in the interface.
var langs = []lang{
	{"中文", "Chinese"},
	{"English", "English"},
	{"日本語", "Japanese"},
	{"한국어", "Korean"},
	{"Français", "French"},
	{"Deutsch", "German"},
	{"Español", "Spanish"},
	{"Русский", "Russian"},
	{"Português", "Portuguese"},
	{"Italiano", "Italian"},
	{"العربية", "Arabic"},
	{"ไทย", "Thai"},
	{"Tiếng Việt", "Vietnamese"},
	{"Bahasa Indonesia", "Indonesian"},
}

// langLabels are the labels of langs, in the same order, for the interface
// to show in a drop-down.
var langLabels = func() []string {
	labels := make([]string, 0, len(langs))
	for _, l := range langs {
		labels = append(labels, l.label)
	}
	return labels
}()

// langName returns the name the model knows a label by. An unknown label
// is passed through, so a language added to the interface without a name
// still reaches the model in some form.
func langName(label string) string {
	for _, l := range langs {
		if l.label == label {
			return l.name
		}
	}
	return label
}

// ollamaHost is where the Ollama server listens.
func ollamaHost() string {
	if host := strings.TrimSpace(os.Getenv(ollamaHostEnv)); host != "" {
		return strings.TrimRight(host, "/")
	}
	return defaultOllamaHost
}

// ollamaModel is the model that does the translating.
func ollamaModel() string {
	if model := strings.TrimSpace(os.Getenv(ollamaModelEnv)); model != "" {
		return model
	}
	return defaultOllamaModel
}

// generateRequest is what Ollama's /api/generate endpoint takes.
type generateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// generateResponse is what the endpoint answers with.
type generateResponse struct {
	Response string `json:"response"`
	Error    string `json:"error"`
}

// ollamaHTTP is the client that talks to the local Ollama server. It
// keeps its connections between requests.
var ollamaHTTP = &http.Client{
	Timeout: translateTimeout,
	// The server is on this machine. The environment sets a proxy for the
	// internet; it must not catch this request.
	Transport: &http.Transport{Proxy: nil},
}

// ollamaClient talks to the Ollama server.
type ollamaClient struct {
	host   string
	model  string
	client *http.Client
}

// newOllamaClient returns a client for the server and model the
// environment names.
func newOllamaClient() *ollamaClient {
	return &ollamaClient{
		host:   ollamaHost(),
		model:  ollamaModel(),
		client: ollamaHTTP,
	}
}

// translate asks the model to translate text from one language into
// another. The prompt names both languages, without which the model
// treats the text as a conversation instead of something to translate.
func (c *ollamaClient) translate(from, to, text string) (string, error) {
	prompt := fmt.Sprintf("Translate from %s to %s: %s", langName(from), langName(to), text)
	body, err := json.Marshal(generateRequest{Model: c.model, Prompt: prompt})
	if err != nil {
		return "", err
	}
	resp, err := c.client.Post(c.host+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("无法连接 Ollama（%s），请确认 ollama serve 已启动：%w", c.host, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("读取 Ollama 响应失败：%w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama 返回 %d：%s", resp.StatusCode, snippet(string(raw)))
	}
	var out generateResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("解析 Ollama 响应失败：%w", err)
	}
	if out.Error != "" {
		return "", errors.New(out.Error)
	}
	result := strings.TrimSpace(out.Response)
	if result == "" {
		return "", errors.New("模型没有返回译文")
	}
	return result, nil
}

// snippet shortens a message that would be too long to show.
func snippet(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > maxErrorBodyLength {
		return s[:maxErrorBodyLength] + "…"
	}
	return s
}

// translateText translates text with the Ollama server this machine runs.
func translateText(from, to, text string) (string, error) {
	return newOllamaClient().translate(from, to, text)
}
