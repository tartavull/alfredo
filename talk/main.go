package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-isatty"
	"github.com/muesli/termenv"

    "github.com/tartavull/alfredo/talk/common"
    "github.com/tartavull/alfredo/talk/ui"
)


func main() {
	renderer := lipgloss.NewRenderer(os.Stderr, termenv.WithColorCache(true))
	opts := []tea.ProgramOption{tea.WithOutput(renderer.Output())}
	if !isatty.IsTerminal(os.Stdin.Fd()) {
		opts = append(opts, tea.WithInput(nil))
	}
	s := common.MakeStyles(renderer)
    c := common.NewCommon(0,0, &s)

	p := tea.NewProgram(ui.New(&c), opts...)
	_, err := p.Run()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
