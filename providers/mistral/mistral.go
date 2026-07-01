// Package mistral provides an HTTP client and message-conversion helpers
// that translate between OpenAI-compatible request/response shapes and the
// Mistral AI chat API format.
package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
)

// ── API key rotation ──────────────────────────────────────────────────────────

var apiKeys = []string{
	"OvyYvnGXctsRF8iKV3ERP9JRCGSectER",
	"0FtPtrOJN2ekL0jKtn66WqB2NYPMgmtR",
	"sRBHaKE7IVu78eWtvmEdiXIQrbsL0fE0",
	"FmDFFaxbTbhqqCZ55mCRGNxIeJdpTTW8",
}

var keyCounter uint64

// NextKey returns the next API key using round-robin, safe for concurrent use.
func NextKey() string {
	idx := atomic.AddUint64(&keyCounter, 1) - 1
	return apiKeys[idx%uint64(len(apiKeys))]
}

// ── Constants ─────────────────────────────────────────────────────────────────

const (
	BaseURL      = "https://api.mistral.ai/v1/chat/completions"
	Model        = "mistral-small-2603"
	DisplayModel = "Mistral-Small-4"
)

// ── Types ─────────────────────────────────────────────────────────────────────

// ToolFunction is the function field inside a tool call.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

// ToolCall is a single tool invocation as understood by both OpenAI and Mistral.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function ToolFunction `json:"function"`
}

// IncomingMessage is an OpenAI-formatted chat message from the client.
type IncomingMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Message is a Mistral-formatted chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolDef is a tool definition (OpenAI / Mistral share this schema).
type ToolDef struct {
	Type     string      `json:"type"`
	Function ToolFuncDef `json:"function"`
}

// ToolFuncDef describes one function tool.
type ToolFuncDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Request is the payload sent to the Mistral API.
type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Tools    []ToolDef `json:"tools,omitempty"`
}

// CompMessage is the message inside a non-streaming Mistral response choice.
type CompMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// CompChoice is a single choice in a non-streaming Mistral response.
type CompChoice struct {
	Index        int         `json:"index"`
	Message      CompMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// CompResponse is a full non-streaming Mistral API response.
type CompResponse struct {
	Choices []CompChoice `json:"choices"`
}

// StreamDelta is the incremental content inside a streaming chunk.
type StreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// StreamChoice is a single choice inside a streaming chunk.
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        StreamDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// StreamChunk is a single server-sent event chunk from the Mistral API.
type StreamChunk struct {
	Choices []StreamChoice `json:"choices"`
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// ContentString coerces an OpenAI content field (string or array) to a plain string.
func ContentString(c any) string {
	if c == nil {
		return ""
	}
	if s, ok := c.(string); ok {
		return s
	}
	b, _ := json.Marshal(c)
	return string(b)
}

// ArgsToString serialises tool-call arguments to a JSON string as OpenAI expects.
func ArgsToString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case nil:
		return "{}"
	default:
		b, _ := json.Marshal(s)
		return string(b)
	}
}

// BuildMessages converts an OpenAI message list to Mistral format, prepending
// the system prompt from mistral_system_prompt.go. If the client supplies a
// system message it takes priority over the default SystemPrompt.
func BuildMessages(inc []IncomingMessage) []Message {
	sysContent := SystemPrompt
	for _, m := range inc {
		if strings.ToLower(m.Role) == "system" {
			if s := ContentString(m.Content); s != "" {
				sysContent = s
			}
			break
		}
	}

	out := []Message{{Role: "system", Content: sysContent}}
	for _, m := range inc {
		role := strings.ToLower(m.Role)
		if role == "system" {
			continue
		}

		if role == "tool" || m.ToolCallID != "" {
			out = append(out, Message{
				Role:       "tool",
				Content:    ContentString(m.Content),
				ToolCallID: m.ToolCallID,
			})
			continue
		}

		content := ContentString(m.Content)
		if role == "assistant" && len(m.ToolCalls) > 0 {
			content = ""
		}

		om := Message{
			Role:    m.Role,
			Content: content,
		}

		if len(m.ToolCalls) > 0 {
			tcs := make([]ToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				otc := ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: ToolFunction{
						Name: tc.Function.Name,
					},
				}
				switch v := tc.Function.Arguments.(type) {
				case string:
					var obj any
					if json.Unmarshal([]byte(v), &obj) == nil {
						otc.Function.Arguments = obj
					} else {
						otc.Function.Arguments = v
					}
				default:
					otc.Function.Arguments = v
				}
				tcs = append(tcs, otc)
			}
			om.ToolCalls = tcs
		}

		out = append(out, om)
	}
	return out
}

// ── HTTP client ───────────────────────────────────────────────────────────────

var httpClient = &http.Client{
	Transport: &http.Transport{DisableCompression: true},
}

// Do sends a single request to the Mistral API and returns the raw HTTP response.
// The caller is responsible for closing resp.Body.
func Do(ctx context.Context, messages []Message, tools []ToolDef, stream bool) (*http.Response, error) {
	payload := Request{
		Model:    Model,
		Messages: messages,
		Stream:   stream,
		Tools:    tools,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("mistral: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, BaseURL, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("mistral: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+NextKey())

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mistral: upstream: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		log.Printf("mistral: upstream error status=%d body=%s", resp.StatusCode, body)
		return nil, fmt.Errorf("mistral: upstream returned %d: %s", resp.StatusCode, body)
	}
	return resp, nil
}
