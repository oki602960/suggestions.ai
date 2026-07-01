// Package ollama provides an HTTP client and message-conversion helpers
// that translate between OpenAI-compatible request/response shapes and the
// Ollama chat API format.
package ollama

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
        "c568998aba6140b0ac3df232f92cfb10.wNpEOzl2OolwK0ZJg2i9nx0d",
        "8359a9f0fa6d42bea558d1c31f323827.6QzYIqvLIK14ZcjTt5MgQOPo",
        "270c3c8be58543cfb38ab668b9535cde.TT--iyVPN4FB9ZOJd6OfApDP",
        "c9630f0a0f124f05a269f39cf459bed3.4AoNV5BYPaDp4Fo9ynx8HM_8",
        "b2a3a59874714d909b4ebe1a5f34d984.1J6VuGyEvYYEbFPQc17ofMfz",
        "cbb3979ed3ac46c387ef67a8ff8d829d.Erm_3Jevlh70hw9avuoMii1A",
        "e76bc5c5c8e040a2a01c19a121a6bc25.Eo8FBKh_twvpwEK70pilbPdu",
        "274d195c9b39438eba39d416defe4060.rEgteKuwLCrVXZWeIUIT13MA",
        "00b7636d96e5484f8076af79fe584765.pRW58NFkszwKb-HNFYDDRRow",
        "4e6d6b7ef0d74fdea7c2d239e5ac8a6b.mgd7TOFUPyYaUmLvBYkHm4xR",
        "4b6033cbbff449c1ac012f28fa93858c.4v8d7oW4TZsSWhTFqzs4WSgs",
        "ff7a764a09a244248e45f5a5193cc2c2.PI6GCv8LdcFStp5hBcDSCgS1",
        "75f001ea95c94950a581cbeb12221646.s3ygTIMiUiI8IOlRtUbujePR",
        "545782757ba242b9ba90ff40fbaad2c2.PfrBiK4Uz3a0pQ-xXb0NEj2i",
}

var keyCounter uint64

// NextKey returns the next API key using round-robin, safe for concurrent use.
func NextKey() string {
        idx := atomic.AddUint64(&keyCounter, 1) - 1
        return apiKeys[idx%uint64(len(apiKeys))]
}

// ── Constants ─────────────────────────────────────────────────────────────────

const (
        BaseURL      = "https://ollama.com/api/chat"
        Model        = "gemma4:31b"
        DisplayModel = "Gemma-4"
)

// ── Types ─────────────────────────────────────────────────────────────────────

// ToolFunction is the function field inside a tool call.
type ToolFunction struct {
        Name      string `json:"name"`
        Arguments any    `json:"arguments,omitempty"`
}

// ToolCall is a single tool invocation as understood by both OpenAI and Ollama.
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

// Message is an Ollama-formatted chat message.
type Message struct {
        Role      string     `json:"role"`
        Content   string     `json:"content"`
        Thinking  string     `json:"thinking,omitempty"`
        ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolDef is a tool definition (OpenAI / Ollama share this schema).
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

// Request is the payload sent to the Ollama API.
type Request struct {
        Model    string    `json:"model"`
        Messages []Message `json:"messages"`
        Stream   bool      `json:"stream"`
        Think    bool      `json:"think"`
        Tools    []ToolDef `json:"tools,omitempty"`
}

// Chunk is a single streamed or non-streamed response chunk from Ollama.
type Chunk struct {
        Message Message `json:"message"`
        Done    bool    `json:"done"`
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

// BuildMessages converts an OpenAI message list to Ollama format, prepending
// the system prompt from ollamaprompt.go.  If the client supplies a system
// message it takes priority over the default SystemPrompt.
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
                                Role:    "tool",
                                Content: ContentString(m.Content),
                        })
                        continue
                }

                content := ContentString(m.Content)
                // Drop thinking/content from assistant messages that have tool_calls
                // so it doesn't pollute the conversation history sent back to Ollama.
                if strings.ToLower(m.Role) == "assistant" && len(m.ToolCalls) > 0 {
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

// Do sends a single request to the Ollama API and returns the raw HTTP response.
// The caller is responsible for closing resp.Body.
func Do(ctx context.Context, messages []Message, tools []ToolDef, stream bool) (*http.Response, error) {
        payload := Request{
                Model:    Model,
                Messages: messages,
                Stream:   stream,
                Think:    false,
                Tools:    tools,
        }

        payloadBytes, err := json.Marshal(payload)
        if err != nil {
                return nil, fmt.Errorf("ollama: marshal request: %w", err)
        }

        req, err := http.NewRequestWithContext(ctx, http.MethodPost, BaseURL, bytes.NewReader(payloadBytes))
        if err != nil {
                return nil, fmt.Errorf("ollama: build request: %w", err)
        }
        req.Header.Set("Content-Type", "application/json")
        req.Header.Set("Authorization", "Bearer "+NextKey())

        resp, err := httpClient.Do(req)
        if err != nil {
                return nil, fmt.Errorf("ollama: upstream: %w", err)
        }
        if resp.StatusCode < 200 || resp.StatusCode >= 300 {
                body, _ := io.ReadAll(resp.Body)
                resp.Body.Close()
                log.Printf("ollama: upstream error status=%d body=%s", resp.StatusCode, body)
                return nil, fmt.Errorf("ollama: upstream returned %d: %s", resp.StatusCode, body)
        }
        return resp, nil
}
