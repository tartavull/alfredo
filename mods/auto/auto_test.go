package auto

import (
	"testing"
)

func TestAddPrompt(t *testing.T) {
    a := New()
    msg := "Hello, world"
    result := a.AddPrompt(msg)
    if len(result) <= len(msg) {
        t.Errorf("Expected '%s'  to be longer than '%s'", result, msg)
    }
}

func TestParseResponse(t *testing.T) {
	auto := New()
	testJson := `{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"question":"Some question"}]
	}`
	err := auto.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseCmd(t *testing.T) {
	auto := New()
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"cmd":"Some cmd"}]
	}`
	err := auto.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseQuestion(t *testing.T) {
	auto := New()
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"question":"Some question"}]
	}`
	err := auto.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseMissingField(t *testing.T) {
	auto := New()
	testJson := `
	{
		"context": "Some context",
		"actions": [{"question":"Some question"}]
	}`
	err := auto.ParseResponse(testJson)
	if err == nil || err.Error() != "missing field 'goal' in JSON" {
		t.Errorf("Expected error due to missing field but got: %v", err)
	}
}

func TestParseExtraField(t *testing.T) {
	auto := New()
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"question":"Some question"}],
		"responses": "Some response"
	}`
	err := auto.ParseResponse(testJson)
	if err == nil || err.Error() != "extra field 'responses' in JSON" {
		t.Errorf("Expected error due to extra field but got: %v", err)
	}
}

func TestParseInvalidAction(t *testing.T) {
	auto := New()
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"invalid":"Some invalid action"}]
	}`
	err := auto.ParseResponse(testJson)
	if err == nil || err.Error() != "actions must contain either a valid 'question' or 'cmd' object" {
		t.Errorf("Expected error due to invalid action but got: %v", err)
	}
}

