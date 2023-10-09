package chat

import (
    "fmt"
	"encoding/json"
    "reflect"
)

type LLMResponse struct {
	Context  string     `json:"context"`
	Goal     string     `json:"goal"`
	Questions []string `json:"questions"`
    Commands []string  `json:"commands"`
}

func ParseResponse(jsonStr string) (*LLMResponse, error) {
    jsonBlob := []byte(jsonStr)

    response := LLMResponse{}
	err := json.Unmarshal([]byte(jsonBlob), &response)

	if err != nil {
		return nil, err
	}

    // Create a map to hold the JSON data
	var data map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonBlob), &data); err != nil {
		return nil, err
	}

	// Check for extra fields
	responseType := reflect.TypeOf(response)
    for key := range data {
		if !structHasField(responseType, key) {
			return nil, fmt.Errorf("extra field '%s' in JSON", key)
		}
	}

    // Check for missing fields
    for i := 0; i < responseType.NumField(); i++ {
		field := responseType.Field(i)
		_, ok := data[field.Tag.Get("json")]
		if !ok {
			return nil, fmt.Errorf("missing field '%s' in JSON", field.Tag.Get("json"))
		}
	}
    return &response, nil
}

// Helper function to check if a JSON field name matches any struct fields
func structHasField(structType reflect.Type, jsonFieldName string) bool {
	for i := 0; i < structType.NumField(); i++ {
		field := structType.Field(i)
		if field.Tag.Get("json") == jsonFieldName {
			return true
		}
	}
	return false
}
