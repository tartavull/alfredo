package auto

import (
	"testing"
    "github.com/charmbracelet/mods/common"
    "github.com/charmbracelet/lipgloss"
)

func TestParseResponse(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `{
		"context": "Some context",
		"goal": "Some goal",
		"questions": ["Some question"],
		"commands": ["Some cmd"]
	}`
	_, err := a.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseMultipleQuestionsAndCommands(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"questions": ["Some question", "Another question"],
		"commands": ["Some cmd", "Another cmd"]
	}`
	_, err := a.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseInvalidQuestionType(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"questions": [1]
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "json: cannot unmarshal number into Go struct field LLMResponse.questions of type string" {
		t.Errorf("Expected error due to invalid question type but got: %v", err)
	}
}

func TestParseInvalidCommandType(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"commands": [1]
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "json: cannot unmarshal number into Go struct field LLMResponse.commands of type string" {
		t.Errorf("Expected error due to invalid command type but got: %v", err)
	}
}

func TestParseMissingField(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"questions": ["Some question"]
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "missing field 'goal' in JSON" {
		t.Errorf("Expected error due to missing field but got: %v", err)
	}
}

func TestParseExtraField(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"questions": ["Some question"],
		"responses": "Some response"
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "extra field 'responses' in JSON" {
		t.Errorf("Expected error due to extra field but got: %v", err)
	}
}

