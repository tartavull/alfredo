package common

import (
	"github.com/charmbracelet/lipgloss"
)

type Styles struct {
	AppName      lipgloss.Style
    App          lipgloss.Style
	CliArgs      lipgloss.Style
	Comment      lipgloss.Style
	CyclingChars lipgloss.Style
	ErrorHeader  lipgloss.Style
	ErrorDetails lipgloss.Style
	Flag         lipgloss.Style
	FlagComma    lipgloss.Style
	FlagDesc     lipgloss.Style
	InlineCode   lipgloss.Style
	Link         lipgloss.Style
	Pipe         lipgloss.Style
	Quote        lipgloss.Style

	Context      lipgloss.Style
	ContextTag   lipgloss.Style
	Goal         lipgloss.Style
	GoalTag      lipgloss.Style
	Question     lipgloss.Style
	QuestionTag  lipgloss.Style

    Command      lipgloss.Style
}

func MakeStyles(r *lipgloss.Renderer) (s Styles) {
	s.AppName = r.NewStyle().Bold(true)
    s.App = r.NewStyle().Margin(1, 2)
	s.CliArgs = r.NewStyle().Foreground(lipgloss.Color("#585858"))
	s.Comment = r.NewStyle().Foreground(lipgloss.Color("#757575"))
	s.CyclingChars = r.NewStyle().Foreground(lipgloss.Color("#FF87D7"))
	s.ErrorHeader = r.NewStyle().Foreground(lipgloss.Color("#F1F1F1")).Background(lipgloss.Color("#FF5F87")).Bold(true).Padding(0, 1).SetString("ERROR")
	s.ErrorDetails = s.Comment.Copy()
	s.Flag = r.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#00B594", Dark: "#3EEFCF"}).Bold(true)
	s.FlagComma = r.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#5DD6C0", Dark: "#427C72"}).SetString(",")
	s.FlagDesc = s.Comment.Copy()
	s.InlineCode = r.NewStyle().Foreground(lipgloss.Color("#FF5F87")).Background(lipgloss.Color("#3A3A3A")).Padding(0, 1)
	s.Link = r.NewStyle().Foreground(lipgloss.Color("#00AF87")).Underline(true)
	s.Quote = r.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#FF71D0", Dark: "#FF78D2"})
	s.Pipe = r.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#8470FF", Dark: "#745CFF"})

    s.Context = s.CliArgs.Copy()
    s.ContextTag = s.Context.Copy().Bold(true).SetString("Context:")

    s.Goal = r.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#00B594", Dark: "#3EEFCF"})
    s.GoalTag = s.Goal.Copy().Bold(true).SetString("Goal:")

    s.Question = s.Quote.Copy()
    s.QuestionTag = s.Question.Copy().Bold(true).SetString("Question:")

	s.Command = r.NewStyle().Foreground(lipgloss.Color("#F1F1F1")).Background(lipgloss.Color("#000000")).Padding(1, 1)
	return s
}
