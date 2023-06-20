package sandbox

import (
	"strings"
	"testing"
	"os"
	"path/filepath"
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

	if res.Stderr == "" || !strings.Contains(res.Stderr, "command not found") {
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

func TestExecuteCommand_Patch(t *testing.T) {
	tempDir := os.TempDir()
	originalFilePath := filepath.Join(tempDir, "original.txt")
	patchFilePath := filepath.Join(tempDir, "patch.txt")

	// Creating a file
	cmd := "echo 'Original content' > " + originalFilePath
	res := ExecuteCommand(cmd)
	if res.ExitCode != 0 || res.Stderr != "" {
		t.Errorf("Failed to create original.txt. ExitCode: %d, Error: %v", res.ExitCode, res.Stderr)
	}

	// Creating a patch
	cmds := []string{
		"echo '--- original.txt' > " + patchFilePath,
		"echo '+++ original.txt' >> " + patchFilePath,
		"echo '@@ -1 +1 @@' >> " + patchFilePath,
		"echo '-Original content' >> " + patchFilePath,
		"echo '+Updated content' >> " + patchFilePath,
	}
	for _, cmd = range cmds {
		res = ExecuteCommand(cmd)
		if res.ExitCode != 0 || res.Stderr != "" {
			t.Errorf("Failed to create/update patch.txt. Command: '%s', ExitCode: %d, Error: %v", cmd, res.ExitCode, res.Stderr)
		}
	}

	// Applying the patch
	cmd = "patch " + originalFilePath + " < " + patchFilePath
	res = ExecuteCommand(cmd)
	if res.ExitCode != 0 || res.Stderr != "" {
		t.Errorf("Failed to apply patch. ExitCode: %d, Error: %v", res.ExitCode, res.Stderr)
	}

	// Verifying the patch
	cmd = "cat " + originalFilePath
	res = ExecuteCommand(cmd)
	if res.ExitCode != 0 || res.Stderr != "" {
		t.Errorf("Failed to read original.txt. ExitCode: %d, Error: %v", res.ExitCode, res.Stderr)
	}
	expectedContent := "Updated content"
	if strings.TrimSpace(res.Stdout) != expectedContent {
		t.Errorf("Expected '%s', got '%s'", expectedContent, res.Stdout)
	}

	// Cleanup
	cmds = []string{
		"rm " + originalFilePath,
		"rm " + patchFilePath,
	}
	for _, cmd = range cmds {
		res = ExecuteCommand(cmd)
		if res.ExitCode != 0 || res.Stderr != "" {
			t.Errorf("Failed to clean up. Command: '%s', ExitCode: %d, Error: %v", cmd, res.ExitCode, res.Stderr)
		}
	}
}
