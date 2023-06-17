package auto 


import (
    "fmt"
	"encoding/json"
    "reflect"
)

type Auto struct {
    Response *LLMResponse
}

func New() *Auto {
    return &Auto{}
}

const InitialPrompt = `
{
    "prompt": "You are part of a program that helps the user to build complex projects. You are able to carry two type of actions: to ask the user clarifying questions such that you don't need to make any assumptions, and to execute commands on the user's terminal.
    Your response should be structured as JSON and include the fields context, goal and actions. See a description of each field below:

    'context': use this field to summarize in a paragraph the result of the previous actions. In other words, you use this field to summarize your current understanding of the situation.
    'goal': given the current context, you must summarize the desire goal that you want to achieve next.
    'actions': this is a list of the type { 'question': "How are you?" } or { 'cmd': 'tree ./ ' }. As exampled, you can either ask questions to the user or run commands in the terminal. You are not allowed to access files outside of the current directory neither directly with your commands or indirectly, for example, by a program you write and execute. When you actions include writting code, there is a strong preference to use Golang.

    Here is a simple example of your response
    { 'context': 'The user wants to understand how long has the current machine being on',
      'goal': 'I will use the program uptime to get extract such information',
      'actions': [{'cmd': 'uptime'},{'question':'is there a particular format in which you want the time to be shown?'}]
    }
    "

    In the current directory is a git repository of the code that parses, executes your responses and feeds them back to you.
    There is a file called plan.md which describes how to make this tool smarter and more useful. Your goal is to help us perfect this plan and also to carry out the plan.
    %s

    "previous": [],
}
`

func (a *Auto) AddPrompt(prompt string) string {
    // FIXME 
    return fmt.Sprintf(InitialPrompt, prompt)
}

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
