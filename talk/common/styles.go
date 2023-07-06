package common

import . "github.com/charmbracelet/lipgloss" //nolint: revive

// Colors.
var (
	normal          = AdaptiveColor{Light: "#1A1A1A", Dark: "#dddddd"}
	normalDim       = AdaptiveColor{Light: "#A49FA5", Dark: "#777777"}
	gray            = AdaptiveColor{Light: "#909090", Dark: "#626262"}
	midGray         = AdaptiveColor{Light: "#B2B2B2", Dark: "#4A4A4A"}
	darkGray        = AdaptiveColor{Light: "#DDDADA", Dark: "#3C3C3C"}
	brightGray      = AdaptiveColor{Light: "#847A85", Dark: "#979797"}
	dimBrightGray   = AdaptiveColor{Light: "#C2B8C2", Dark: "#4D4D4D"}
	indigo          = AdaptiveColor{Light: "#5A56E0", Dark: "#7571F9"}
	dimIndigo       = AdaptiveColor{Light: "#9498FF", Dark: "#494690"}
	subtleIndigo    = AdaptiveColor{Light: "#7D79F6", Dark: "#514DC1"}
	dimSubtleIndigo = AdaptiveColor{Light: "#BBBDFF", Dark: "#383584"}
	cream           = AdaptiveColor{Light: "#FFFDF5", Dark: "#FFFDF5"}
	yellowGreen     = AdaptiveColor{Light: "#04B575", Dark: "#ECFD65"}
	dullYellowGreen = AdaptiveColor{Light: "#6BCB94", Dark: "#9BA92F"}
	fuchsia         = AdaptiveColor{Light: "#EE6FF8", Dark: "#EE6FF8"}
	dimFuchsia      = AdaptiveColor{Light: "#F1A8FF", Dark: "#99519E"}
	dullFuchsia     = AdaptiveColor{Dark: "#AD58B4", Light: "#F793FF"}
	dimDullFuchsia  = AdaptiveColor{Light: "#F6C9FF", Dark: "#6B3A6F"}
	green           = Color("#04B575")
	red             = AdaptiveColor{Light: "#FF4672", Dark: "#ED567A"}
	faintRed        = AdaptiveColor{Light: "#FF6F91", Dark: "#C74665"}

	semiDimGreen = AdaptiveColor{Light: "#35D79C", Dark: "#036B46"}
	dimGreen     = AdaptiveColor{Light: "#72D2B0", Dark: "#0B5137"}
)

type Styles struct {
	AppName      Style
    App          Style
	CliArgs      Style
	Comment      Style
	CyclingChars Style
	ErrorHeader  Style
	ErrorDetails Style
	Flag         Style
	FlagComma    Style
	FlagDesc     Style
	InlineCode   Style
	Link         Style
	Pipe         Style
	Quote        Style

    Logo         Style
	Context      Style
	ContextTag   Style
	Goal         Style
	GoalTag      Style
	Question     Style
	QuestionTag  Style

    Command      Style
}

func MakeStyles(r *Renderer) (s Styles) {
	s.AppName = r.NewStyle().Bold(true)
    s.App = r.NewStyle().Margin(1, 2)
	s.CliArgs = r.NewStyle().Foreground(Color("#585858"))
	s.Comment = r.NewStyle().Foreground(Color("#757575"))
	s.CyclingChars = r.NewStyle().Foreground(Color("#FF87D7"))
	s.ErrorHeader = r.NewStyle().Foreground(Color("#F1F1F1")).Background(Color("#FF5F87")).Bold(true).Padding(0, 1).SetString("ERROR")
	s.ErrorDetails = s.Comment.Copy()
	s.Flag = r.NewStyle().Foreground(AdaptiveColor{Light: "#00B594", Dark: "#3EEFCF"}).Bold(true)
	s.FlagComma = r.NewStyle().Foreground(AdaptiveColor{Light: "#5DD6C0", Dark: "#427C72"}).SetString(",")
	s.FlagDesc = s.Comment.Copy()
	s.InlineCode = r.NewStyle().Foreground(Color("#FF5F87")).Background(Color("#3A3A3A")).Padding(0, 1)
	s.Link = r.NewStyle().Foreground(Color("#00AF87")).Underline(true)
	s.Quote = r.NewStyle().Foreground(AdaptiveColor{Light: "#FF71D0", Dark: "#FF78D2"})
	s.Pipe = r.NewStyle().Foreground(AdaptiveColor{Light: "#8470FF", Dark: "#745CFF"})

    s.Logo  = r.NewStyle().
        Foreground(Color("#ECFD65")).
        Background(indigo).
        Bold(true)

    s.Context = s.CliArgs.Copy()
    s.ContextTag = s.Context.Copy().Bold(true).SetString("Context:")

    s.Goal = r.NewStyle().Foreground(AdaptiveColor{Light: "#00B594", Dark: "#3EEFCF"})
    s.GoalTag = s.Goal.Copy().Bold(true).SetString("Goal:")

    s.Question = s.Quote.Copy()
    s.QuestionTag = s.Question.Copy().Bold(true).SetString("Question:")

	s.Command = r.NewStyle().Foreground(Color("#F1F1F1")).Background(Color("#000000")).Padding(1, 1)
	return s
}
