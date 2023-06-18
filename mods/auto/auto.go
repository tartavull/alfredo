package auto 

import (
    "fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/bubbles/textarea"
    "github.com/charmbracelet/bubbles/cursor"

    "github.com/charmbracelet/mods/common"
    "github.com/charmbracelet/mods/sandbox"
)

type State int

const (
    stateAnswering State = iota
    stateCompleted
)

type Auto struct {
    styles common.Styles 
    textarea textarea.Model
    state State
    Response *LLMResponse
    answers []string
    outputs []sandbox.Result
}

func New(s common.Styles) *Auto {
    a := &Auto{
        textarea: textarea.New(),
		styles:   s,
    }

	a.textarea.Placeholder = ">"
    a.textarea.ShowLineNumbers = false
    a.textarea.Focus()

    testJson := `
	{
		"context": "Some context",
		"goal": "Some goal",
		"questions": ["Some question"],
        "commands": ["cat plan.md", "another action"]
	}`
	a.Response, _ = a.ParseResponse(testJson)
    a.state = stateAnswering
    a.answers = []string{}
    a.outputs = []sandbox.Result{}

    return a 
}

func (a *Auto) Focus() tea.Cmd {
    return textarea.Blink
}

func (a *Auto) SetSize(width, height int) {
    a.textarea.SetWidth(width)
    a.textarea.MaxHeight = height / 2
}


func (a *Auto) Update(msg tea.Msg) (*Auto, tea.Cmd) {
    var cmd tea.Cmd
    cmds := make([]tea.Cmd, 0)

	if a.state == stateAnswering {
    	switch msg := msg.(type) {
        case tea.KeyMsg:
			if len(a.outputs) < len(a.Response.Commands) {
				if msg.String() == "y" {
					for _, cmd := range a.Response.Commands {
						a.outputs = append(a.outputs, sandbox.ExecuteCommand(cmd))
					}
				}
			} else if len(a.answers) < len(a.Response.Questions) {
				a.textarea, cmd = a.textarea.Update(msg)
				cmds = append(cmds, cmd)
				if msg.String() == "ctrl+d" {
					a.answers = append(a.answers, a.textarea.Value())
					a.textarea.Reset()
				}
            } else {
				a.state = stateCompleted
            }
        case cursor.BlinkMsg:
        	cmds = append(cmds, textarea.Blink)
        }
    }
    return a, tea.Batch(cmds...) 
}

func (a *Auto) View() string {
    view := ""

    view += a.styles.ContextTag.String() + " " + a.styles.Context.Render(a.Response.Context)
    view += "\n\n"
    view += a.styles.GoalTag.String() + " " + a.styles.Goal.Render(a.Response.Goal)
    view += "\n\n"
    if a.state == stateAnswering {
        if len(a.outputs) < len(a.Response.Commands) {
			var styledCmds []string
			for _, cmd := range a.Response.Commands {
				styledCmds = append(styledCmds, fmt.Sprintf("$ %s", cmd))
			}
			styledCmdStr := strings.Join(styledCmds, "\n")
			view += a.styles.Command.Render(styledCmdStr) + "\n\n"
            view += a.styles.Comment.Render("Press y to execute all commands or esc to exit")
        } else if len(a.answers) < len(a.Response.Questions) {
            view += a.styles.QuestionTag.String() + " " + a.styles.Question.Render(a.Response.Questions[len(a.answers)])
            view += "\n\n"
            view += a.textarea.View()
            view += "\n"
            view += a.styles.Comment.Render(fmt.Sprintf("Question %d of %d: press ctrl+d to submit answer", len(a.answers)+1, len(a.Response.Questions)))
        }
    } else if a.state == stateCompleted {
        view += fmt.Sprintf("%+v\n", a.answers)
        view += fmt.Sprintf("%+v\n", a.outputs)
    }

    return a.styles.App.Render(view)
}
