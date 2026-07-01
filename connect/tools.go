package connect

import "openai-ollama-proxy/providers/ollama"

// getRegisteredToolDefs returns built-in tool definitions that are always
// appended to every request's tool list.  Add your own tools here.
func getRegisteredToolDefs() []ollama.ToolDef {
	return nil
}
