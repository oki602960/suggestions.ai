package main

import (
	"fmt"
	"net/http"
	"os"

	"openai-ollama-proxy/connect"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "5000"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", connect.Connect)
	mux.HandleFunc("/v1/models", connect.Models)

	fmt.Printf("OpenAI-Ollama proxy listening on :%s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}
