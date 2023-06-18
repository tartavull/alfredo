package auto

import (
    "fmt"
	"encoding/json"
    "reflect"
)

type LLMResponse struct {
	Context string  `json:"context"`
	Goal    string  `json:"goal"`
	Actions []Action `json:"actions"`
}

type Action interface {
	IsValid() bool
}

type Question struct {
	Question string `json:"question"`
}

type Cmd struct {
	Cmd string `json:"cmd"`
}

func (q Question) IsValid() bool {
	return q.Question != ""
}

func (c Cmd) IsValid() bool {
	return c.Cmd != ""
}

func (c *LLMResponse) UnmarshalJSON(data []byte) error {
	type Alias LLMResponse
	aux := &struct {
		Actions []json.RawMessage `json:"actions"`
		*Alias
	}{
		Alias: (*Alias)(c),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	for _, action := range aux.Actions {
		var q Question
		if err := json.Unmarshal(action, &q); err == nil && q.IsValid() {
			c.Actions = append(c.Actions, q)
			continue
		}
		var cmd Cmd
		if err := json.Unmarshal(action, &cmd); err == nil && cmd.IsValid() {
			c.Actions = append(c.Actions, cmd)
			continue
		}
		return fmt.Errorf("actions must contain either a valid 'question' or 'cmd' object")
	}
	return nil
}

func (a *Auto) ParseResponse(jsonStr string) error {
    jsonBlob := []byte(jsonStr)

    response := LLMResponse{}
	err := json.Unmarshal([]byte(jsonBlob), &response)

	if err != nil {
		return err
	}

    // Create a map to hold the JSON data
	var data map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonBlob), &data); err != nil {
		return err
	}

	// Check for extra fields
	responseType := reflect.TypeOf(response)
    for key := range data {
		if !structHasField(responseType, key) {
			return fmt.Errorf("extra field '%s' in JSON", key)
		}
	}

    // Check for missing fields
    for i := 0; i < responseType.NumField(); i++ {
		field := responseType.Field(i)
		_, ok := data[field.Tag.Get("json")]
		if !ok {
			return fmt.Errorf("missing field '%s' in JSON", field.Tag.Get("json"))
		}
	}
	
    a.Response = &response
    return nil
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
