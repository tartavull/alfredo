package sandbox

import (
	"bytes"
	"os/exec"
	"strings"
)

type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

func ExecuteCommand(cmd string) Result {
	var result Result

	// Basic safety validation
	if strings.Contains(cmd, "sudo") {
        result.Stderr = "execution of 'sudo' command is not allowed"
		return result
	}

    command := exec.Command("bash", "-c", cmd)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr

    err := command.Run()

	// Capturing the exit code
	if exitError, ok := err.(*exec.ExitError); ok {
		result.ExitCode = exitError.ExitCode()
	}

	// Save command output
	result.Stdout = stdout.String()
	result.Stderr = stderr.String()

	return result
}

