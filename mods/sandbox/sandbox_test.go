package sandbox

import (
	"strings"
	"testing"
)

func TestExecuteCommand_Success(t *testing.T) {
	cmd := "echo hello"
	res := ExecuteCommand(cmd)

	if res.Stderr != "" {
		t.Errorf("Expected no Stderr, got: %v", res.Stderr)
	}

	if res.ExitCode != 0 {
		t.Errorf("Expected exit code 0, got: %d", res.ExitCode)
	}

	if strings.TrimSpace(res.Stdout) != "hello" {
		t.Errorf("Expected 'hello', got: '%s'", res.Stdout)
	}
}

func TestExecuteCommand_Failure(t *testing.T) {
    cmd := "nonexistentcommand"
	res := ExecuteCommand(cmd)

	if res.Stderr == "" || !strings.Contains(res.Stderr, "does not exist") {
		t.Errorf("Expected error, got %v", res.Stderr)
	}
}

func TestExecuteCommand_Security(t *testing.T) {
	cmd := "sudo echo hello"
	res := ExecuteCommand(cmd)

	if res.Stderr == "" || !strings.Contains(res.Stderr, "not allowed") {
		t.Errorf("Expected an error about 'sudo' not being allowed, got: %v", res.Stderr)
	}
}

