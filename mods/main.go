package main

import (
	"fmt"
	"os"
	"runtime/debug"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glow/editor"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-isatty"
	"github.com/muesli/termenv"
	flag "github.com/spf13/pflag"
)

// Build vars.
var (
	//nolint: gochecknoglobals
	version = "dev"
	commit  = ""
	date    = ""
	builtBy = ""
)

func buildVersion() string {
	result := "mods version " + version
	if commit != "" {
		result = fmt.Sprintf("%s\ncommit: %s", result, commit)
	}
	if date != "" {
		result = fmt.Sprintf("%s\nbuilt at: %s", result, date)
	}
	if builtBy != "" {
		result = fmt.Sprintf("%s\nbuilt by: %s", result, builtBy)
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Sum != "" {
		result = fmt.Sprintf("%s\nmodule version: %s, checksum: %s", result, info.Main.Version, info.Main.Sum)
	}
	return result
}

func main() {
	renderer := lipgloss.NewRenderer(os.Stderr, termenv.WithColorCache(true))
	opts := []tea.ProgramOption{tea.WithOutput(renderer.Output())}
	if !isatty.IsTerminal(os.Stdin.Fd()) {
		opts = append(opts, tea.WithInput(nil))
	}
	mods := newMods(renderer)
	p := tea.NewProgram(mods, opts...)
	m, err := p.Run()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	mods = m.(*Mods)
	if mods.Error != nil {
		os.Exit(1)
	}
	if mods.Config.Settings {
		c := editor.Cmd(mods.Config.SettingsPath)
		c.Stdin = os.Stdin
		c.Stdout = os.Stdout
		c.Stderr = os.Stderr
		if err := c.Run(); err != nil {
			mods.Error = &modsError{reason: "Missing $EDITOR", err: err}
			fmt.Println(mods.ErrorView())
			os.Exit(1)
		}
		fmt.Println("Wrote config file to:", mods.Config.SettingsPath)
		os.Exit(0)
	}
	if mods.Config.Version {
		fmt.Println(buildVersion())
		os.Exit(0)
	}
    if mods.Config.Auto {
        // Display a text edit?
    }
	if mods.Config.ShowHelp || (mods.Input == "" && mods.Config.Prefix == "") {
		flag.Usage()
		os.Exit(0)
	}
	fmt.Println(mods.FormattedOutput())
}
