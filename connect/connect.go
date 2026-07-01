// Service connect implements an OpenAI-compatible SSE streaming endpoint
// backed by the Ollama and Mistral providers.
package connect

import (
        "bufio"
        "encoding/json"
        "fmt"
        "io"
        "net/http"
        "strings"
        "time"

        "openai-ollama-proxy/providers/mistral"
        "openai-ollama-proxy/providers/ollama"
        "openai-ollama-proxy/providers/openrouter"
        "openai-ollama-proxy/providers/sambanova"
)

// ── OpenAI-compatible types ─────────────────────────────────────────────────────────────

type incomingRequest struct {
        Model      string                   `json:"model,omitempty"`
        Messages   []ollama.IncomingMessage `json:"messages"`
        Tools      []ollama.ToolDef         `json:"tools,omitempty"`
        ToolChoice any                      `json:"tool_choice,omitempty"`
        Stream     *bool                    `json:"stream,omitempty"`
}

type streamingToolCall struct {
        Index    int                 `json:"index"`
        ID       string              `json:"id,omitempty"`
        Type     string              `json:"type,omitempty"`
        Function ollama.ToolFunction `json:"function"`
}

type delta struct {
        Role             string              `json:"role,omitempty"`
        Content          string              `json:"content,omitempty"`
        ReasoningContent string              `json:"reasoning_content,omitempty"`
        ToolCalls        []streamingToolCall `json:"tool_calls,omitempty"`
}

type chunkChoice struct {
        Index        int     `json:"index"`
        Delta        delta   `json:"delta"`
        FinishReason *string `json:"finish_reason"`
}

type openAIChunk struct {
        ID      string        `json:"id"`
        Object  string        `json:"object"`
        Created int64         `json:"created"`
        Model   string        `json:"model"`
        Choices []chunkChoice `json:"choices"`
}

type nonStreamChoice struct {
        Index        int            `json:"index"`
        Message      ollama.Message `json:"message"`
        FinishReason string         `json:"finish_reason"`
}

type nonStreamResponse struct {
        ID      string            `json:"id"`
        Object  string            `json:"object"`
        Created int64             `json:"created"`
        Model   string            `json:"model"`
        Choices []nonStreamChoice `json:"choices"`
        Usage   struct {
                PromptTokens     int `json:"prompt_tokens"`
                CompletionTokens int `json:"completion_tokens"`
                TotalTokens      int `json:"total_tokens"`
        } `json:"usage"`
}

// ── Helpers ──────────────────────────────────────────────────────────────────────────────

func toMistralMessages(inc []ollama.IncomingMessage) []mistral.IncomingMessage {
        out := make([]mistral.IncomingMessage, len(inc))
        for i, m := range inc {
                out[i] = mistral.IncomingMessage{
                        Role:       m.Role,
                        Content:    m.Content,
                        ToolCallID: m.ToolCallID,
                }
                if len(m.ToolCalls) > 0 {
                        tcs := make([]mistral.ToolCall, len(m.ToolCalls))
                        for j, tc := range m.ToolCalls {
                                tcs[j] = mistral.ToolCall{
                                        ID:   tc.ID,
                                        Type: tc.Type,
                                        Function: mistral.ToolFunction{
                                                Name:      tc.Function.Name,
                                                Arguments: tc.Function.Arguments,
                                        },
                                }
                        }
                        out[i].ToolCalls = tcs
                }
        }
        return out
}

func toMistralToolDefs(tds []ollama.ToolDef) []mistral.ToolDef {
        out := make([]mistral.ToolDef, len(tds))
        for i, td := range tds {
                out[i] = mistral.ToolDef{
                        Type: td.Type,
                        Function: mistral.ToolFuncDef{
                                Name:        td.Function.Name,
                                Description: td.Function.Description,
                                Parameters:  td.Function.Parameters,
                        },
                }
        }
        return out
}

func toOpenRouterMessages(inc []ollama.IncomingMessage) []openrouter.IncomingMessage {
        out := make([]openrouter.IncomingMessage, len(inc))
        for i, m := range inc {
                out[i] = openrouter.IncomingMessage{
                        Role:       m.Role,
                        Content:    m.Content,
                        ToolCallID: m.ToolCallID,
                }
                if len(m.ToolCalls) > 0 {
                        tcs := make([]openrouter.ToolCall, len(m.ToolCalls))
                        for j, tc := range m.ToolCalls {
                                tcs[j] = openrouter.ToolCall{
                                        ID:   tc.ID,
                                        Type: tc.Type,
                                        Function: openrouter.ToolFunction{
                                                Name:      tc.Function.Name,
                                                Arguments: tc.Function.Arguments,
                                        },
                                }
                        }
                        out[i].ToolCalls = tcs
                }
        }
        return out
}

func toOpenRouterToolDefs(tds []ollama.ToolDef) []openrouter.ToolDef {
        out := make([]openrouter.ToolDef, len(tds))
        for i, td := range tds {
                out[i] = openrouter.ToolDef{
                        Type: td.Type,
                        Function: openrouter.ToolFuncDef{
                                Name:        td.Function.Name,
                                Description: td.Function.Description,
                                Parameters:  td.Function.Parameters,
                        },
                }
        }
        return out
}

func toSambaNovaMessages(inc []ollama.IncomingMessage) []sambanova.IncomingMessage {
        out := make([]sambanova.IncomingMessage, len(inc))
        for i, m := range inc {
                out[i] = sambanova.IncomingMessage{
                        Role:       m.Role,
                        Content:    m.Content,
                        ToolCallID: m.ToolCallID,
                }
                if len(m.ToolCalls) > 0 {
                        tcs := make([]sambanova.ToolCall, len(m.ToolCalls))
                        for j, tc := range m.ToolCalls {
                                tcs[j] = sambanova.ToolCall{
                                        ID:   tc.ID,
                                        Type: tc.Type,
                                        Function: sambanova.ToolFunction{
                                                Name:      tc.Function.Name,
                                                Arguments: tc.Function.Arguments,
                                        },
                                }
                        }
                        out[i].ToolCalls = tcs
                }
        }
        return out
}

func toSambaNovaToolDefs(tds []ollama.ToolDef) []sambanova.ToolDef {
        out := make([]sambanova.ToolDef, len(tds))
        for i, td := range tds {
                out[i] = sambanova.ToolDef{
                        Type: td.Type,
                        Function: sambanova.ToolFuncDef{
                                Name:        td.Function.Name,
                                Description: td.Function.Description,
                                Parameters:  td.Function.Parameters,
                        },
                }
        }
        return out
}

// ── Handler ──────────────────────────────────────────────────────────────────────────

// Connect is the OpenAI-compatible chat completions endpoint.
//
//encore:api public raw path=/v1/chat/completions
func Connect(w http.ResponseWriter, req *http.Request) {
        if req.Method == http.MethodOptions {
                w.Header().Set("Access-Control-Allow-Origin", "*")
                w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
                w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
                w.WriteHeader(http.StatusNoContent)
                return
        }
        if req.Method != http.MethodPost {
                http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
                return
        }

        body, err := io.ReadAll(req.Body)
        if err != nil {
                http.Error(w, "Bad Request", http.StatusBadRequest)
                return
        }

        var incoming incomingRequest
        if err := json.Unmarshal(body, &incoming); err != nil {
                http.Error(w, "Invalid JSON", http.StatusBadRequest)
                return
        }

        isMistral := strings.HasPrefix(strings.ToLower(incoming.Model), "mistral")
        isOpenRouter := strings.HasPrefix(strings.ToLower(incoming.Model), "tencent")
        isSambaNova := strings.HasPrefix(strings.ToLower(incoming.Model), "deepseek")

        allTools := append(incoming.Tools, getRegisteredToolDefs()...)
        hasTools := len(allTools) > 0
        wantStream := incoming.Stream != nil && *incoming.Stream

        var displayName string
        if isMistral {
                displayName = mistral.DisplayModel
        } else if isOpenRouter {
                displayName = openrouter.DisplayModel
        } else if isSambaNova {
                displayName = sambanova.DisplayModel
        } else {
                displayName = ollama.DisplayModel
        }

        // ── Non-streaming ──────────────────────────────────────────────────────────────────
        if !wantStream {
                created := time.Now().Unix()
                finishReason := "stop"
                msg := ollama.Message{Role: "assistant"}

                if isMistral {
                        messages := mistral.BuildMessages(toMistralMessages(incoming.Messages))
                        mistralTools := toMistralToolDefs(allTools)
                        resp, err := mistral.Do(req.Context(), messages, mistralTools, false)
                        if err != nil {
                                http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr mistral.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                http.Error(w, "Bad upstream response: "+err.Error(), http.StatusBadGateway)
                                return
                        }

                        upstream := mr.Choices[0].Message
                        msg.Content = upstream.Content
                        if len(upstream.ToolCalls) > 0 {
                                finishReason = "tool_calls"
                                msg.Content = ""
                                tcs := make([]ollama.ToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, ollama.ToolCall{
                                                ID:   fmt.Sprintf("call_%d_%d", created, i),
                                                Type: "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: mistral.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                msg.ToolCalls = tcs
                        }
                } else if isOpenRouter {
                        messages := openrouter.BuildMessages(toOpenRouterMessages(incoming.Messages))
                        openRouterTools := toOpenRouterToolDefs(allTools)
                        resp, err := openrouter.Do(req.Context(), displayName, messages, openRouterTools, false)
                        if err != nil {
                                http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr openrouter.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                http.Error(w, "Bad upstream response: "+err.Error(), http.StatusBadGateway)
                                return
                        }

                        upstream := mr.Choices[0].Message
                        msg.Content = upstream.Content
                        msg.Thinking = upstream.ReasoningContent
                        if msg.Thinking == "" {
                                msg.Thinking = upstream.Reasoning
                        }
                        if len(upstream.ToolCalls) > 0 {
                                finishReason = "tool_calls"
                                msg.Content = ""
                                tcs := make([]ollama.ToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, ollama.ToolCall{
                                                ID:   fmt.Sprintf("call_%d_%d", created, i),
                                                Type: "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: openrouter.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                msg.ToolCalls = tcs
                        }
                } else if isSambaNova {
                        messages := sambanova.BuildMessages(toSambaNovaMessages(incoming.Messages))
                        sambaTools := toSambaNovaToolDefs(allTools)
                        resp, err := sambanova.Do(req.Context(), displayName, messages, sambaTools, false)
                        if err != nil {
                                http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr sambanova.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                http.Error(w, "Bad upstream response: "+err.Error(), http.StatusBadGateway)
                                return
                        }

                        upstream := mr.Choices[0].Message
                        msg.Content = upstream.Content
                        msg.Thinking = upstream.ReasoningContent
                        if msg.Thinking == "" {
                                msg.Thinking = upstream.Reasoning
                        }
                        if len(upstream.ToolCalls) > 0 {
                                finishReason = "tool_calls"
                                msg.Content = ""
                                tcs := make([]ollama.ToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, ollama.ToolCall{
                                                ID:   fmt.Sprintf("call_%d_%d", created, i),
                                                Type: "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: sambanova.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                msg.ToolCalls = tcs
                        }
                } else {
                        messages := ollama.BuildMessages(incoming.Messages)
                        resp, err := ollama.Do(req.Context(), messages, allTools, false)
                        if err != nil {
                                http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var ollamaResp ollama.Chunk
                        if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
                                http.Error(w, "Bad upstream response: "+err.Error(), http.StatusBadGateway)
                                return
                        }

                        msg.Content = ollamaResp.Message.Content
                        if len(ollamaResp.Message.ToolCalls) > 0 {
                                finishReason = "tool_calls"
                                msg.Content = ""
                                tcs := make([]ollama.ToolCall, 0, len(ollamaResp.Message.ToolCalls))
                                for i, tc := range ollamaResp.Message.ToolCalls {
                                        tcs = append(tcs, ollama.ToolCall{
                                                ID:   fmt.Sprintf("call_%d_%d", created, i),
                                                Type: "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: ollama.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                msg.ToolCalls = tcs
                        }
                }

                nr := nonStreamResponse{
                        ID:      fmt.Sprintf("chatcmpl-%d", created),
                        Object:  "chat.completion",
                        Created: created,
                        Model:   displayName,
                        Choices: []nonStreamChoice{{Index: 0, Message: msg, FinishReason: finishReason}},
                }
                w.Header().Set("Content-Type", "application/json")
                w.Header().Set("Access-Control-Allow-Origin", "*")
                json.NewEncoder(w).Encode(nr)
                return
        }

        // ── Streaming ───────────────────────────────────────────────────────────────────────────
        w.Header().Set("Content-Type", "text/event-stream")
        w.Header().Set("Cache-Control", "no-cache, no-transform")
        w.Header().Set("Connection", "keep-alive")
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("X-Accel-Buffering", "no")
        w.WriteHeader(http.StatusOK)

        flusher, canFlush := w.(http.Flusher)
        created := time.Now().Unix()
        id := fmt.Sprintf("chatcmpl-%d", created)

        writeSSE := func(d delta, finishReason *string) {
                chunk := openAIChunk{
                        ID:      id,
                        Object:  "chat.completion.chunk",
                        Created: created,
                        Model:   displayName,
                        Choices: []chunkChoice{{Index: 0, Delta: d, FinishReason: finishReason}},
                }
                b, _ := json.Marshal(chunk)
                fmt.Fprintf(w, "data: %s\n\n", b)
                if canFlush {
                        flusher.Flush()
                }
        }

        flushDone := func() {
                fmt.Fprintf(w, "data: [DONE]\n\n")
                if canFlush {
                        flusher.Flush()
                }
        }

        writeSSE(delta{Role: "assistant"}, nil)

        if hasTools {
                if isMistral {
                        messages := mistral.BuildMessages(toMistralMessages(incoming.Messages))
                        mistralTools := toMistralToolDefs(allTools)
                        resp, err := mistral.Do(req.Context(), messages, mistralTools, false)
                        if err != nil {
                                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr mistral.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                fmt.Fprintf(w, "data: {\"error\":\"bad upstream response\"}\n\n")
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }

                        upstream := mr.Choices[0].Message
                        if len(upstream.ToolCalls) > 0 {
                                tcs := make([]streamingToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, streamingToolCall{
                                                Index: i,
                                                ID:    fmt.Sprintf("call_%d_%d", created, i),
                                                Type:  "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: mistral.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                writeSSE(delta{ToolCalls: tcs}, nil)
                                reason := "tool_calls"
                                writeSSE(delta{}, &reason)
                        } else {
                                if upstream.Content != "" {
                                        writeSSE(delta{Content: upstream.Content}, nil)
                                }
                                stop := "stop"
                                writeSSE(delta{}, &stop)
                        }
                        flushDone()
                        return
                }

                if isOpenRouter {
                        messages := openrouter.BuildMessages(toOpenRouterMessages(incoming.Messages))
                        openRouterTools := toOpenRouterToolDefs(allTools)
                        resp, err := openrouter.Do(req.Context(), displayName, messages, openRouterTools, false)
                        if err != nil {
                                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr openrouter.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                fmt.Fprintf(w, "data: {\"error\":\"bad upstream response\"}\n\n")
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }

                        upstream := mr.Choices[0].Message
                        if len(upstream.ToolCalls) > 0 {
                                tcs := make([]streamingToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, streamingToolCall{
                                                Index: i,
                                                ID:    fmt.Sprintf("call_%d_%d", created, i),
                                                Type:  "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: openrouter.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                writeSSE(delta{ToolCalls: tcs}, nil)
                                reason := "tool_calls"
                                writeSSE(delta{}, &reason)
                        } else {
                                if upstream.Content != "" {
                                        writeSSE(delta{Content: upstream.Content}, nil)
                                }
                                reasoning := upstream.ReasoningContent
                                if reasoning == "" {
                                        reasoning = upstream.Reasoning
                                }
                                if reasoning != "" {
                                        writeSSE(delta{ReasoningContent: reasoning}, nil)
                                }
                                stop := "stop"
                                writeSSE(delta{}, &stop)
                        }
                        flushDone()
                        return
                }

                if isSambaNova {
                        messages := sambanova.BuildMessages(toSambaNovaMessages(incoming.Messages))
                        sambaTools := toSambaNovaToolDefs(allTools)
                        resp, err := sambanova.Do(req.Context(), displayName, messages, sambaTools, false)
                        if err != nil {
                                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }
                        defer resp.Body.Close()

                        respBody, _ := io.ReadAll(resp.Body)
                        var mr sambanova.CompResponse
                        if err := json.Unmarshal(respBody, &mr); err != nil || len(mr.Choices) == 0 {
                                fmt.Fprintf(w, "data: {\"error\":\"bad upstream response\"}\n\n")
                                if canFlush {
                                        flusher.Flush()
                                }
                                return
                        }

                        upstream := mr.Choices[0].Message
                        if len(upstream.ToolCalls) > 0 {
                                tcs := make([]streamingToolCall, 0, len(upstream.ToolCalls))
                                for i, tc := range upstream.ToolCalls {
                                        tcs = append(tcs, streamingToolCall{
                                                Index: i,
                                                ID:    fmt.Sprintf("call_%d_%d", created, i),
                                                Type:  "function",
                                                Function: ollama.ToolFunction{
                                                        Name:      tc.Function.Name,
                                                        Arguments: sambanova.ArgsToString(tc.Function.Arguments),
                                                },
                                        })
                                }
                                writeSSE(delta{ToolCalls: tcs}, nil)
                                reason := "tool_calls"
                                writeSSE(delta{}, &reason)
                        } else {
                                if upstream.Content != "" {
                                        writeSSE(delta{Content: upstream.Content}, nil)
                                }
                                reasoning := upstream.ReasoningContent
                                if reasoning == "" {
                                        reasoning = upstream.Reasoning
                                }
                                if reasoning != "" {
                                        writeSSE(delta{ReasoningContent: reasoning}, nil)
                                }
                                stop := "stop"
                                writeSSE(delta{}, &stop)
                        }
                        flushDone()
                        return
                }

                // Ollama tools path (existing)
                messages := ollama.BuildMessages(incoming.Messages)
                resp, err := ollama.Do(req.Context(), messages, allTools, false)
                if err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                        if canFlush {
                                flusher.Flush()
                        }
                        return
                }
                defer resp.Body.Close()

                respBody, _ := io.ReadAll(resp.Body)
                var ollamaResp ollama.Chunk
                if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"bad upstream response\"}\n\n")
                        if canFlush {
                                flusher.Flush()
                        }
                        return
                }

                if len(ollamaResp.Message.ToolCalls) > 0 {
                        tcs := make([]streamingToolCall, 0, len(ollamaResp.Message.ToolCalls))
                        for i, tc := range ollamaResp.Message.ToolCalls {
                                tcs = append(tcs, streamingToolCall{
                                        Index: i,
                                        ID:    fmt.Sprintf("call_%d_%d", created, i),
                                        Type:  "function",
                                        Function: ollama.ToolFunction{
                                                Name:      tc.Function.Name,
                                                Arguments: ollama.ArgsToString(tc.Function.Arguments),
                                        },
                                })
                        }
                        writeSSE(delta{ToolCalls: tcs}, nil)
                        reason := "tool_calls"
                        writeSSE(delta{}, &reason)
                } else {
                        if ollamaResp.Message.Content != "" {
                                writeSSE(delta{Content: ollamaResp.Message.Content}, nil)
                        }
                        if ollamaResp.Message.Thinking != "" {
                                writeSSE(delta{ReasoningContent: ollamaResp.Message.Thinking}, nil)
                        }
                        stop := "stop"
                        writeSSE(delta{}, &stop)
                }
                flushDone()
                return
        }

        // No tools — real-time streaming
        if isMistral {
                messages := mistral.BuildMessages(toMistralMessages(incoming.Messages))
                mistralTools := toMistralToolDefs(allTools)
                resp, err := mistral.Do(req.Context(), messages, mistralTools, true)
                if err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                        if canFlush {
                                flusher.Flush()
                        }
                        return
                }
                defer resp.Body.Close()

                reader := bufio.NewReader(resp.Body)
                for {
                        line, readErr := reader.ReadString('\n')
                        line = strings.TrimSpace(line)

                        if strings.HasPrefix(line, "data: ") {
                                data := strings.TrimPrefix(line, "data: ")
                                if data == "[DONE]" {
                                        flushDone()
                                        return
                                }
                                var chunk mistral.StreamChunk
                                if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
                                        choice := chunk.Choices[0]
                                        if choice.FinishReason != nil {
                                                writeSSE(delta{}, choice.FinishReason)
                                                flushDone()
                                                return
                                        }
                                        d := delta{}
                                        if choice.Delta.Role != "" {
                                                d.Role = choice.Delta.Role
                                        }
                                        if choice.Delta.Content != "" {
                                                d.Content = choice.Delta.Content
                                        }
                                        if len(choice.Delta.ToolCalls) > 0 {
                                                tcs := make([]streamingToolCall, 0, len(choice.Delta.ToolCalls))
                                                for i, tc := range choice.Delta.ToolCalls {
                                                        tcs = append(tcs, streamingToolCall{
                                                                Index: i,
                                                                ID:    tc.ID,
                                                                Type:  tc.Type,
                                                                Function: ollama.ToolFunction{
                                                                        Name:      tc.Function.Name,
                                                                        Arguments: mistral.ArgsToString(tc.Function.Arguments),
                                                                },
                                                        })
                                                }
                                                d.ToolCalls = tcs
                                        }
                                        if d.Role != "" || d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0 {
                                                writeSSE(d, nil)
                                        }
                                }
                        }

                        if readErr == io.EOF || readErr != nil {
                                break
                        }
                }
                flushDone()
                return
        }

        if isOpenRouter {
                messages := openrouter.BuildMessages(toOpenRouterMessages(incoming.Messages))
                openRouterTools := toOpenRouterToolDefs(allTools)
                resp, err := openrouter.Do(req.Context(), displayName, messages, openRouterTools, true)
                if err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                        if canFlush {
                                flusher.Flush()
                        }
                        return
                }
                defer resp.Body.Close()

                reader := bufio.NewReader(resp.Body)
                for {
                        line, readErr := reader.ReadString('\n')
                        line = strings.TrimSpace(line)

                        if strings.HasPrefix(line, "data: ") {
                                data := strings.TrimPrefix(line, "data: ")
                                if data == "[DONE]" {
                                        flushDone()
                                        return
                                }
                                var chunk openrouter.StreamChunk
                                if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
                                        choice := chunk.Choices[0]
                                        if choice.FinishReason != nil {
                                                writeSSE(delta{}, choice.FinishReason)
                                                flushDone()
                                                return
                                        }
                                        d := delta{}
                                        if choice.Delta.Role != "" {
                                                d.Role = choice.Delta.Role
                                        }
                                        if choice.Delta.Content != "" {
                                                d.Content = choice.Delta.Content
                                        }
                                        if choice.Delta.ReasoningContent != "" {
                                                d.ReasoningContent = choice.Delta.ReasoningContent
                                        } else if choice.Delta.Reasoning != "" {
                                                d.ReasoningContent = choice.Delta.Reasoning
                                        }
                                        if len(choice.Delta.ToolCalls) > 0 {
                                                tcs := make([]streamingToolCall, 0, len(choice.Delta.ToolCalls))
                                                for i, tc := range choice.Delta.ToolCalls {
                                                        tcs = append(tcs, streamingToolCall{
                                                                Index: i,
                                                                ID:    tc.ID,
                                                                Type:  tc.Type,
                                                                Function: ollama.ToolFunction{
                                                                        Name:      tc.Function.Name,
                                                                        Arguments: openrouter.ArgsToString(tc.Function.Arguments),
                                                                },
                                                        })
                                                }
                                                d.ToolCalls = tcs
                                        }
                                        if d.Role != "" || d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0 {
                                                writeSSE(d, nil)
                                        }
                                }
                        }

                        if readErr == io.EOF || readErr != nil {
                                break
                        }
                }
                flushDone()
                return
        }

        if isSambaNova {
                messages := sambanova.BuildMessages(toSambaNovaMessages(incoming.Messages))
                sambaTools := toSambaNovaToolDefs(allTools)
                resp, err := sambanova.Do(req.Context(), displayName, messages, sambaTools, true)
                if err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                        if canFlush {
                                flusher.Flush()
                        }
                        return
                }
                defer resp.Body.Close()

                reader := bufio.NewReader(resp.Body)
                for {
                        line, readErr := reader.ReadString('\n')
                        line = strings.TrimSpace(line)

                        if strings.HasPrefix(line, "data: ") {
                                data := strings.TrimPrefix(line, "data: ")
                                if data == "[DONE]" {
                                        flushDone()
                                        return
                                }
                                var chunk sambanova.StreamChunk
                                if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
                                        choice := chunk.Choices[0]
                                        if choice.FinishReason != nil {
                                                writeSSE(delta{}, choice.FinishReason)
                                                flushDone()
                                                return
                                        }
                                        d := delta{}
                                        if choice.Delta.Role != "" {
                                                d.Role = choice.Delta.Role
                                        }
                                        if choice.Delta.Content != "" {
                                                d.Content = choice.Delta.Content
                                        }
                                        if choice.Delta.ReasoningContent != "" {
                                                d.ReasoningContent = choice.Delta.ReasoningContent
                                        } else if choice.Delta.Reasoning != "" {
                                                d.ReasoningContent = choice.Delta.Reasoning
                                        }
                                        if len(choice.Delta.ToolCalls) > 0 {
                                                tcs := make([]streamingToolCall, 0, len(choice.Delta.ToolCalls))
                                                for i, tc := range choice.Delta.ToolCalls {
                                                        tcs = append(tcs, streamingToolCall{
                                                                Index: i,
                                                                ID:    tc.ID,
                                                                Type:  tc.Type,
                                                                Function: ollama.ToolFunction{
                                                                        Name:      tc.Function.Name,
                                                                        Arguments: sambanova.ArgsToString(tc.Function.Arguments),
                                                                },
                                                        })
                                                }
                                                d.ToolCalls = tcs
                                        }
                                        if d.Role != "" || d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0 {
                                                writeSSE(d, nil)
                                        }
                                }
                        }

                        if readErr == io.EOF || readErr != nil {
                                break
                        }
                }
                flushDone()
                return
        }

        // Ollama no-tools streaming (existing)
        messages := ollama.BuildMessages(incoming.Messages)
        resp, err := ollama.Do(req.Context(), messages, allTools, true)
        if err != nil {
                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                if canFlush {
                        flusher.Flush()
                }
                return
        }
        defer resp.Body.Close()

        reader := bufio.NewReader(resp.Body)
        for {
                line, readErr := reader.ReadString('\n')
                line = strings.TrimSpace(line)

                if line != "" {
                        var chunk ollama.Chunk
                        if json.Unmarshal([]byte(line), &chunk) == nil {
                                if chunk.Done {
                                        stop := "stop"
                                        writeSSE(delta{}, &stop)
                                        flushDone()
                                        return
                                }
                                if chunk.Message.Content != "" {
                                        writeSSE(delta{Content: chunk.Message.Content}, nil)
                                }
                                if chunk.Message.Thinking != "" {
                                        writeSSE(delta{ReasoningContent: chunk.Message.Thinking}, nil)
                                }
                        }
                }

                if readErr == io.EOF || readErr != nil {
                        break
                }
        }
        flushDone()
}

// ── Models list endpoint ───────────────────────────────────────────────────────────

type modelObject struct {
        ID      string `json:"id"`
        Object  string `json:"object"`
        Created int64  `json:"created"`
        OwnedBy string `json:"owned_by"`
}

type modelsResponse struct {
        Object string        `json:"object"`
        Data   []modelObject `json:"data"`
}

// Models returns the list of available models in OpenAI format.
//
//encore:api public raw path=/v1/models
func Models(w http.ResponseWriter, req *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Header().Set("Access-Control-Allow-Origin", "*")
        resp := modelsResponse{
                Object: "list",
                Data: []modelObject{
                        {
                                ID:      ollama.DisplayModel,
                                Object:  "model",
                                Created: 1700000000,
                                OwnedBy: "ollama",
                        },
                        {
                                ID:      mistral.DisplayModel,
                                Object:  "model",
                                Created: 1700000001,
                                OwnedBy: "mistral",
                        },
                        {
                                ID:      openrouter.DisplayModel,
                                Object:  "model",
                                Created: 1700000002,
                                OwnedBy: "openrouter",
                        },
                        {
                                ID:      sambanova.DisplayModel,
                                Object:  "model",
                                Created: 1700000003,
                                OwnedBy: "sambanova",
                        },
                },
        }
        json.NewEncoder(w).Encode(resp)
}
