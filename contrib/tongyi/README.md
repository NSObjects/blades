# Tongyi Qwen Integration

[![Go Version](https://img.shields.io/badge/Go-1.19+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![API Status](https://img.shields.io/badge/API-OpenAI%20Compatible-orange.svg)](https://dashscope.aliyuncs.com/)

This package provides seamless integration support for Alibaba Cloud's Tongyi Qwen large language models through OpenAI-compatible API interface, enabling you to leverage powerful AI capabilities in your Go applications.

## ✨ Features

- 🚀 **Dual Response Modes**: Support for both streaming and non-streaming chat completion
- 🔧 **Tool Calling**: Full support for Function Calling with recursive tool execution
- 🎯 **Multimodal Input**: Handle text, images, audio, and other content types
- 🔌 **OpenAI Compatible**: Fully compatible with OpenAI API format for easy migration
- 🔑 **Flexible Authentication**: Support for API key parameter passing and environment variables
- 📝 **Predefined Constants**: Ready-to-use model name constants for better maintainability
- ⚡ **High Performance**: Optimized for production use with proper error handling
- 🧪 **Well Tested**: Comprehensive test suite with unit and integration tests

## 🎯 Supported Models

| Model | Constant | Description | Use Case |
|-------|----------|-------------|----------|
| **QwenTurbo** | `qwen-turbo` | Balanced performance and cost | General purpose, high-frequency requests |
| **QwenPlus** | `qwen-plus` | Enhanced understanding capability | Complex reasoning, detailed analysis |
| **QwenMax** | `qwen-max` | Highest performance | Critical tasks, maximum accuracy |
| **QwenLong** | `qwen-long` | Supports long text processing | Document analysis, long-form content |
| **QwenVL** | `qwen-vl-plus` | Vision-language model | Image understanding, multimodal tasks |
| **QwenAudio** | `qwen-audio-turbo` | Audio processing model | Speech recognition, audio analysis |

```go
// Available model constants
const (
    QwenTurbo  = "qwen-turbo"     // Balanced performance and cost
    QwenPlus   = "qwen-plus"      // Enhanced understanding capability  
    QwenMax    = "qwen-max"       // Highest performance
    QwenLong   = "qwen-long"      // Supports long text
    QwenVL     = "qwen-vl-plus"   // Vision-language model
    QwenAudio  = "qwen-audio-turbo" // Audio model
)
```

## 🚀 Quick Start

### Installation

```bash
go get github.com/go-kratos/blades/contrib/tongyi
```

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/go-kratos/blades"
    "github.com/go-kratos/blades/contrib/tongyi"
)

func main() {
    // Create agent with environment variable API key
    agent := blades.NewAgent(
        "Tongyi Assistant",
        blades.WithModel(tongyi.QwenTurbo),
        blades.WithProvider(tongyi.NewChatProvider()), // Reads from DASHSCOPE_API_KEY
        blades.WithInstructions("You are a helpful assistant."),
    )
    
    // Create a simple prompt
    prompt := blades.NewPrompt(
        blades.UserMessage("Hello, please introduce yourself."),
    )
    
    // Get response
    result, err := agent.Run(context.Background(), prompt)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(result.AsText())
}
```

## 📖 Usage Examples

### 1. Environment Variable Configuration

```bash
# Set your API key
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

```go
// Uses environment variable automatically
agent := blades.NewAgent(
    "Tongyi Agent",
    blades.WithModel(tongyi.QwenPlus),
    blades.WithProvider(tongyi.NewChatProvider()),
)
```

### 2. Direct API Key Passing

```go
// Pass API key directly
agent := blades.NewAgent(
    "Tongyi Agent", 
    blades.WithModel(tongyi.QwenMax),
    blades.WithProvider(tongyi.NewChatProvider("sk-your-api-key")),
)
```

### 3. Streaming Response

```go
// Get streaming response for real-time output
stream, err := agent.RunStream(context.Background(), prompt)
if err != nil {
    log.Fatal(err)
}

for stream.Next() {
    response, err := stream.Current()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(response.AsText()) // Real-time output
}
```

### 4. Multi-turn Conversation

```go
// Create conversation with history
conversation := []*blades.Message{
    {Role: blades.RoleUser, Parts: []blades.Part{blades.TextPart{Text: "Hello"}}},
    {Role: blades.RoleAssistant, Parts: []blades.Part{blades.TextPart{Text: "Hi there!"}}},
    {Role: blades.RoleUser, Parts: []blades.Part{blades.TextPart{Text: "How are you?"}}},
}

prompt := blades.NewPrompt(conversation...)
result, err := agent.Run(context.Background(), prompt)
```

### 5. Tool Calling (Function Calling)

```go
import "github.com/google/jsonschema-go/jsonschema"

// Define a weather tool
weatherTool := &blades.Tool{
    Name:        "get_weather",
    Description: "Get current weather information for a city",
    InputSchema: &jsonschema.Schema{
        Type: "object",
        Properties: map[string]*jsonschema.Schema{
            "city": {
                Type:        "string",
                Description: "The city name to get weather for",
            },
        },
        Required: []string{"city"},
    },
    Handle: func(ctx context.Context, args string) (string, error) {
        // Parse args and implement weather logic
        return "Beijing: Sunny, 25°C", nil
    },
}

// Create agent with tools
agent := blades.NewAgent(
    "Weather Assistant",
    blades.WithModel(tongyi.QwenPlus),
    blades.WithProvider(tongyi.NewChatProvider()),
    blades.WithTools(weatherTool),
)

// Use the tool
prompt := blades.NewPrompt(
    blades.UserMessage("What's the weather like in Beijing?"),
)
result, err := agent.Run(context.Background(), prompt)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Primary API key (recommended)
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Fallback to OpenAI API key format
export OPENAI_API_KEY="your-api-key"
```

### Programmatic Configuration

```go
// Method 1: Direct API key
provider := tongyi.NewChatProvider("sk-your-api-key")

// Method 2: From configuration
apiKey := config.GetString("tongyi.api_key")
provider := tongyi.NewChatProvider(apiKey)

// Method 3: Environment variable (automatic)
provider := tongyi.NewChatProvider() // Reads from DASHSCOPE_API_KEY
```

## 🔧 Advanced Usage

### Model Options

```go
// Use different models for different tasks
agent := blades.NewAgent(
    "Multi-Model Agent",
    blades.WithModel(tongyi.QwenMax), // Highest accuracy
    blades.WithProvider(tongyi.NewChatProvider()),
    blades.WithTemperature(0.7),      // Control creativity
    blades.WithMaxOutputTokens(1000), // Limit response length
)
```

### Error Handling

```go
result, err := agent.Run(ctx, prompt)
if err != nil {
    switch {
    case errors.Is(err, tongyi.ErrInvalidAPIKey):
        log.Fatal("Invalid API key")
    case errors.Is(err, tongyi.ErrInvalidModel):
        log.Fatal("Unsupported model")
    case errors.Is(err, tongyi.ErrEmptyResponse):
        log.Println("No response received")
    default:
        log.Printf("Unexpected error: %v", err)
    }
}
```

## 🧪 Testing

### Run Tests

```bash
# Unit tests
go test ./contrib/tongyi/...

# Integration tests (requires API key)
DASHSCOPE_API_KEY="your-key" go test -v ./contrib/tongyi/...

# With coverage
go test -cover ./contrib/tongyi/...
```

### Example Programs

```bash
# Run example programs
cd examples/tongyi

# Simple chat demo
go run simple_chat.go

# Interactive chat
go run chat_demo.go

# Quick test
go run interactive_demo.go
```

## 📊 Performance

- **Response Time**: 1-3 seconds for typical requests
- **Throughput**: Supports high-frequency requests
- **Memory**: Optimized for low memory usage
- **Concurrency**: Thread-safe, supports concurrent requests

## 🔗 API Reference

### Endpoints

- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Chat Completion**: `/chat/completions`
- **Streaming**: `/chat/completions` (with `stream=true`)

### Rate Limits

- **QwenTurbo**: 1000 requests/minute
- **QwenPlus**: 500 requests/minute  
- **QwenMax**: 200 requests/minute

## ⚠️ Important Notes

1. **API Key Security**: Never commit API keys to version control
2. **Rate Limits**: Respect API rate limits to avoid throttling
3. **Model Selection**: Choose appropriate model based on your use case
4. **Error Handling**: Always implement proper error handling
5. **Context Timeout**: Set appropriate timeouts for production use

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Blades Documentation](https://github.com/go-kratos/blades)
- **Issues**: [GitHub Issues](https://github.com/go-kratos/blades/issues)
- **Discussions**: [GitHub Discussions](https://github.com/go-kratos/blades/discussions)
