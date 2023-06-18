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

	// Splitting the command string into name and args
	segments := strings.Fields(cmd)
	name := segments[0]
	args := segments[1:]

    // Check if the program exists
	_, err := exec.LookPath(name)
	if err != nil {
		result.Stderr = "program '" + name + "' does not exist or is not in PATH"
		return result
	}

	command := exec.Command(name, args...)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr

	err = command.Run()

	// Capturing the exit code
	if exitError, ok := err.(*exec.ExitError); ok {
		result.ExitCode = exitError.ExitCode()
	}

	// Save command output
	result.Stdout = stdout.String()
	result.Stderr = stderr.String()

	return result
}

