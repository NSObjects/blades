package main

import (
	"context"
	"log"
	"os"

	"github.com/go-kratos/blades"
	"github.com/go-kratos/blades/contrib/tongyi"
)

func main() {
	// Method 1: Using environment variable DASHSCOPE_API_KEY
	agent1 := blades.NewAgent(
		"Tongyi Chat Agent (Env)",
		blades.WithModel(tongyi.QwenTurbo),            // Using constant
		blades.WithProvider(tongyi.NewChatProvider()), // Read API key from environment variable
		blades.WithInstructions("You are a helpful assistant that provides detailed and accurate information."),
	)

	// Method 2: Passing API key directly
	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set DASHSCOPE_API_KEY environment variable")
	}

	agent2 := blades.NewAgent(
		"Tongyi Chat Agent (Direct)",
		blades.WithModel(tongyi.QwenPlus),                   // Using different model
		blades.WithProvider(tongyi.NewChatProvider(apiKey)), // Pass API key directly
		blades.WithInstructions("You are a professional assistant that can provide accurate and useful information."),
	)

	// Test first agent
	prompt1 := blades.NewPrompt(
		blades.UserMessage("What is the capital of France?"),
	)
	result1, err := agent1.Run(context.Background(), prompt1)
	if err != nil {
		log.Fatal("Agent1 error:", err)
	}
	log.Println("Agent1 (QwenTurbo):", result1.AsText())

	// Test second agent
	prompt2 := blades.NewPrompt(
		blades.UserMessage("Please explain the development history of artificial intelligence."),
	)
	result2, err := agent2.Run(context.Background(), prompt2)
	if err != nil {
		log.Fatal("Agent2 error:", err)
	}
	log.Println("Agent2 (QwenPlus):", result2.AsText())
}
