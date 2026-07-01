# OpenAI-compatible Ollama Proxy — Encore

A drop-in OpenAI API proxy backed by Ollama, ready to deploy on [Encore](https://encore.dev).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible streaming + non-streaming chat |
| `GET`  | `/v1/models`           | Returns the available model list |

Both endpoints send `Access-Control-Allow-Origin: *` and handle CORS preflight.

## Project structure

```
encore-app/
├── encore.app                   # Encore app config
├── go.mod
├── connect/
│   ├── connect.go               # Encore service — API handlers
│   └── tools.go                 # Built-in tool registry (extend here)
└── providers/
    └── ollama/
        ├── ollama.go            # HTTP client, types, message conversion
        └── ollamaprompt.go      # ← Edit the default system prompt here
```

## Changing the system prompt

Open `providers/ollama/ollamaprompt.go` and edit the `SystemPrompt` variable:

```go
var SystemPrompt = `You are a helpful AI assistant specialised in ...`
```

Clients can still override the prompt per-request by including a `{"role":"system","content":"..."}` message — the client value takes priority.

## Adding built-in tools

Add `ToolDef` entries to the slice returned by `getRegisteredToolDefs()` in `connect/tools.go`:

```go
func getRegisteredToolDefs() []ollama.ToolDef {
    return []ollama.ToolDef{
        {
            Type: "function",
            Function: ollama.ToolFuncDef{
                Name:        "get_weather",
                Description: "Returns current weather for a city",
                Parameters: map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "city": map[string]any{"type": "string"},
                    },
                    "required": []string{"city"},
                },
            },
        },
    }
}
```

## Deploy on Encore

```bash
# Install Encore CLI
curl -L https://encore.dev/install.sh | bash

# From the encore-app directory:
cd encore-app
encore run          # local dev
encore deploy       # production deploy
```

## Model & API keys

The active model and API key pool are configured in `providers/ollama/ollama.go`:

- `Model` — the Ollama model slug (`gemma4:31b` by default)
- `DisplayModel` — the name returned to OpenAI clients
- `apiKeys` — round-robin key rotation (safe for concurrent requests)
