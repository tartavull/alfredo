package auto

import (
	"testing"
    "github.com/charmbracelet/mods/common"
    "github.com/charmbracelet/lipgloss"
)

func TestAddPrompt(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
    msg := "Hello, world"
    result := a.AddPrompt(msg)
    if len(result) <= len(msg) {
        t.Errorf("Expected '%s'  to be longer than '%s'", result, msg)
    }
}

func TestParseResponse(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"question":"Some question"}]
	}`
	_, err := a.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseCmd(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"cmd":"Some cmd"}]
	}`
	_, err := a.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseQuestion(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"question":"Some question"}]
	}`
	_, err := a.ParseResponse(testJson)
	if err != nil {
		t.Errorf("Error parsing JSON: %v", err)
	}
}

func TestParseMissingField(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"actions": [{"question":"Some question"}]
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
		"actions": [{"question":"Some question"}],
		"responses": "Some response"
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "extra field 'responses' in JSON" {
		t.Errorf("Expected error due to extra field but got: %v", err)
	}
}

func TestParseInvalidAction(t *testing.T) {
	r := lipgloss.DefaultRenderer()
    s := common.MakeStyles(r)
    a := New(s)
	testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"actions": [{"invalid":"Some invalid action"}]
	}`
	_, err := a.ParseResponse(testJson)
	if err == nil || err.Error() != "actions must contain either a valid 'question' or 'cmd' object" {
		t.Errorf("Expected error due to invalid action but got: %v", err)
	}
}

