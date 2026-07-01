---
name: Provider wiring pattern in connect.go
description: Where a new AI provider must be hooked into the proxy's request router.
---

This Go proxy (`connect/connect.go`) dispatches an incoming OpenAI-style chat
request to one of several backend providers (ollama, mistral, openrouter,
sambanova, ...) based on a prefix match against the requested model name.

To add a provider, every one of these spots needs an `isXxx` branch added:
1. Prefix detection (`isXxx := strings.HasPrefix(strings.ToLower(incoming.Model), "prefix")`)
   and the `displayName` if/else chain.
2. Non-streaming response branch.
3. Streaming branch with tools (`hasTools` block).
4. Streaming branch without tools (real-time streaming block).
5. The `/v1/models` endpoint's static list.

Each provider package mirrors the same shape: `IncomingMessage`, `ToolCall`,
`Message`/`CompMessage`, `StreamChunk`/`StreamDelta`, `BuildMessages`, `Do()`,
plus `toXxxMessages`/`toXxxToolDefs` converter helpers in connect.go.

**Why:** the router has no shared abstraction/interface — each provider's
wiring is duplicated by hand across these 5 spots, so missing one silently
breaks that code path only (e.g. streaming works but non-streaming 404s).

**How to apply:** when adding or modifying a provider, search connect.go for
all existing providers' `isXxx` usages to find every spot that needs the new
branch — don't assume one match is enough.
