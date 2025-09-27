package tongyi

import (
	"context"
	"testing"

	"github.com/go-kratos/blades"
	"github.com/google/jsonschema-go/jsonschema"
)

func TestNewChatProvider(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   []string
		wantErr  bool
	}{
		{
			name:    "valid API key",
			apiKey:  []string{"sk-12345678901234567890123456789012"},
			wantErr: false,
		},
		{
			name:    "empty API key",
			apiKey:  []string{""},
			wantErr: false, // Should fallback to environment variable
		},
		{
			name:    "no API key provided",
			apiKey:  []string{},
			wantErr: false, // Should fallback to environment variable
		},
		{
			name:    "invalid API key format",
			apiKey:  []string{"invalid"},
			wantErr: false, // Provider created but will fail on use
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewChatProvider(tt.apiKey...)
			if provider == nil {
				t.Errorf("NewChatProvider() returned nil")
			}
		})
	}
}

func TestIsValidAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   string
		expected bool
	}{
		{
			name:     "valid API key",
			apiKey:   "sk-12345678901234567890123456789012",
			expected: true,
		},
		{
			name:     "empty API key",
			apiKey:   "",
			expected: false,
		},
		{
			name:     "short API key",
			apiKey:   "sk-123",
			expected: false,
		},
		{
			name:     "long valid API key",
			apiKey:   "sk-123456789012345678901234567890123456789012345678901234567890",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidAPIKey(tt.apiKey)
			if result != tt.expected {
				t.Errorf("isValidAPIKey(%q) = %v, want %v", tt.apiKey, result, tt.expected)
			}
		})
	}
}

func TestIsValidModel(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{
			name:     "valid QwenTurbo",
			model:    QwenTurbo,
			expected: true,
		},
		{
			name:     "valid QwenPlus",
			model:    QwenPlus,
			expected: true,
		},
		{
			name:     "valid QwenMax",
			model:    QwenMax,
			expected: true,
		},
		{
			name:     "valid QwenLong",
			model:    QwenLong,
			expected: true,
		},
		{
			name:     "valid QwenVL",
			model:    QwenVL,
			expected: true,
		},
		{
			name:     "valid QwenAudio",
			model:    QwenAudio,
			expected: true,
		},
		{
			name:     "invalid model",
			model:    "invalid-model",
			expected: false,
		},
		{
			name:     "empty model",
			model:    "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidModel(tt.model)
			if result != tt.expected {
				t.Errorf("isValidModel(%q) = %v, want %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestToChatCompletionParams(t *testing.T) {
	tests := []struct {
		name    string
		request *blades.ModelRequest
		options blades.ModelOptions
		wantErr bool
	}{
		{
			name: "valid request with QwenTurbo",
			request: &blades.ModelRequest{
				Model: QwenTurbo,
				Messages: []*blades.Message{
					{
						Role:  blades.RoleUser,
						Parts: []blades.Part{blades.TextPart{Text: "Hello"}},
					},
				},
			},
			options: blades.ModelOptions{},
			wantErr: false,
		},
		{
			name: "invalid model",
			request: &blades.ModelRequest{
				Model: "invalid-model",
				Messages: []*blades.Message{
					{
						Role:  blades.RoleUser,
						Parts: []blades.Part{blades.TextPart{Text: "Hello"}},
					},
				},
			},
			options: blades.ModelOptions{},
			wantErr: true,
		},
		{
			name: "empty messages",
			request: &blades.ModelRequest{
				Model:    QwenTurbo,
				Messages: []*blades.Message{},
			},
			options: blades.ModelOptions{},
			wantErr: true,
		},
		{
			name: "nil messages",
			request: &blades.ModelRequest{
				Model:    QwenTurbo,
				Messages: nil,
			},
			options: blades.ModelOptions{},
			wantErr: true,
		},
		{
			name: "request with options",
			request: &blades.ModelRequest{
				Model: QwenPlus,
				Messages: []*blades.Message{
					{
						Role:  blades.RoleUser,
						Parts: []blades.Part{blades.TextPart{Text: "Hello"}},
					},
				},
			},
			options: blades.ModelOptions{
				Temperature:     0.7,
				TopP:           0.9,
				MaxOutputTokens: 1000,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := toChatCompletionParams(tt.request, tt.options)
			if (err != nil) != tt.wantErr {
				t.Errorf("toChatCompletionParams() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestToTools(t *testing.T) {
	tests := []struct {
		name    string
		tools   []*blades.Tool
		wantErr bool
	}{
		{
			name:    "empty tools",
			tools:   []*blades.Tool{},
			wantErr: false,
		},
		{
			name:    "nil tools",
			tools:   nil,
			wantErr: false,
		},
		{
			name: "valid tool",
			tools: []*blades.Tool{
				{
					Name:        "test_tool",
					Description: "A test tool",
					InputSchema: &jsonschema.Schema{
						Type: "object",
					},
					Handle: func(ctx context.Context, args string) (string, error) {
						return "test result", nil
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool with valid JSON schema",
			tools: []*blades.Tool{
				{
					Name:        "test_tool",
					Description: "A test tool",
					InputSchema: &jsonschema.Schema{
						Type: "object",
					},
					Handle: func(ctx context.Context, args string) (string, error) {
						return "test result", nil
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := toTools(tt.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("toTools() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestToTextParts(t *testing.T) {
	tests := []struct {
		name     string
		message  *blades.Message
		expected int
	}{
		{
			name: "message with text parts",
			message: &blades.Message{
				Parts: []blades.Part{
					blades.TextPart{Text: "Hello"},
					blades.TextPart{Text: "World"},
				},
			},
			expected: 2,
		},
		{
			name: "message with mixed parts",
			message: &blades.Message{
				Parts: []blades.Part{
					blades.TextPart{Text: "Hello"},
					blades.FilePart{Name: "test.txt", URI: "file://test.txt"},
				},
			},
			expected: 1, // Only text parts should be included
		},
		{
			name: "message with no text parts",
			message: &blades.Message{
				Parts: []blades.Part{
					blades.FilePart{Name: "test.txt", URI: "file://test.txt"},
				},
			},
			expected: 0,
		},
		{
			name: "empty message",
			message: &blades.Message{
				Parts: []blades.Part{},
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := toTextParts(tt.message)
			if len(result) != tt.expected {
				t.Errorf("toTextParts() returned %d parts, want %d", len(result), tt.expected)
			}
		})
	}
}

func TestToolCall(t *testing.T) {
	tools := []*blades.Tool{
		{
			Name: "test_tool",
			Handle: func(ctx context.Context, args string) (string, error) {
				return "test result", nil
			},
		},
		{
			Name: "another_tool",
			Handle: func(ctx context.Context, args string) (string, error) {
				return "another result", nil
			},
		},
	}

	tests := []struct {
		name        string
		toolName    string
		args        string
		expected    string
		expectedErr error
	}{
		{
			name:        "valid tool call",
			toolName:    "test_tool",
			args:        "test args",
			expected:    "test result",
			expectedErr: nil,
		},
		{
			name:        "another valid tool call",
			toolName:    "another_tool",
			args:        "test args",
			expected:    "another result",
			expectedErr: nil,
		},
		{
			name:        "tool not found",
			toolName:    "nonexistent_tool",
			args:        "test args",
			expected:    "",
			expectedErr: ErrToolNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := toolCall(context.Background(), tools, tt.toolName, tt.args)
			if err != tt.expectedErr {
				t.Errorf("toolCall() error = %v, wantErr %v", err, tt.expectedErr)
			}
			if result != tt.expected {
				t.Errorf("toolCall() result = %v, want %v", result, tt.expected)
			}
		})
	}
}
