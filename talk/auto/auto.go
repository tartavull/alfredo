package auto

import (
    "log"
	"strings"
    "os"

	tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/bubbles/textarea"
	"github.com/mitchellh/go-wordwrap"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"

    "github.com/tartavull/alfredo/talk/common"
    "github.com/tartavull/alfredo/talk/sandbox"
    "github.com/tartavull/alfredo/talk/components/tabs"
)

type State int

const (
    stateAnswering State = iota
    stateCompleted
)

type Auto struct {
    common *common.Common
    styles common.Styles
    picked pane
    tabs *tabs.Tabs
	panes  []tea.Model

    textarea textarea.Model
    state State
    Response *LLMResponse
    answers []string
    outputs []sandbox.Result

    viewport   viewport.Model
	width      int
	height     int
}

func New(c *common.Common, s common.Styles) *Auto {
    a := &Auto{
        common: c,
        textarea: textarea.New(),
		styles:   s,
        picked: paneChat,
        viewport: viewport.New(0, 0),
		tabs:   InitTabs(c),
        panes: []tea.Model{
            nil,
            nil,
        },
    }
	a.viewport.YPosition = 0

	a.textarea.Placeholder = ">"
    a.textarea.ShowLineNumbers = false
    a.textarea.Focus()

    data, err := os.ReadFile("chat/llm_1.json")
    if err != nil {
        log.Fatal(err)
    }
	a.Response, _ = a.ParseResponse(string(data))
    a.state = stateAnswering
    a.answers = []string{}
    a.outputs = []sandbox.Result{}

    return a
}

func (a *Auto) Init() tea.Cmd {
    return nil
}

type pane int

const (
	paneChat pane = iota
	paneHistory
	paneLast
)

func (p pane) String() string {
	return []string{
		"Chat",
		"History",
	}[p]
}

func InitTabs(c *common.Common) *tabs.Tabs {
	ts := make([]string, paneLast)
	for i, b := range []pane{paneChat, paneHistory} {
		ts[i] = b.String()
	}
	t := tabs.New(c, ts)
	return t
}

func (a *Auto) Focus() tea.Cmd {
    return textarea.Blink
}

func (a *Auto) SetSize(width, height int) {
    a.common.SetSize(width, height)
    a.textarea.SetWidth(width)
    a.textarea.MaxHeight = height / 2
    a.viewport.Width = width
    a.viewport.Height = height
}

func (a *Auto) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmds := make([]tea.Cmd, 0)
	switch msg := msg.(type) {
	case tabs.ActiveTabMsg:
		a.picked = pane(msg)
	}
	model, cmd := a.tabs.Update(msg)
	cmds = append(cmds, cmd)
	a.tabs = model.(*tabs.Tabs)

    /*
	model, cmd = a.panes[a.picked].Update(msg)
	cmds = append(cmds, cmd)
	a.panes[a.picked] = model
    */
	return a, tea.Batch(cmds...)
}

/*
func (a *Auto) Update(msg tea.Msg) (*Auto, tea.Cmd) {
    var cmd tea.Cmd
    cmds := make([]tea.Cmd, 0)

    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        a.width = msg.Width
        a.height = msg.Height
    }

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
    if a.state == stateCompleted {
        prompt := a.buildPrompt()
        fmt.Println(prompt)
    }
    return a, tea.Batch(cmds...)
}
*/

func (a *Auto) wrap(in string) string {
    return wordwrap.WrapString(in, uint(a.textarea.Width()/2))
}

func (a *Auto) View() string {
    var builder strings.Builder
    
    builder.WriteString(a.styles.Logo.Render(" Alfredo "))
    builder.WriteString("\n\n")
    builder.WriteString(a.tabs.View())
    builder.WriteString("\n\n")
    builder.WriteString(a.styles.ContextTag.String() + " " + a.styles.Context.Render(a.wrap(a.Response.Context)))
    /*
    builder.WriteString("\n\n")
    builder.WriteString(a.styles.GoalTag.String() + " " + a.styles.Goal.Render(a.wrap(a.Response.Goal)))
    builder.WriteString("\n\n")
    
    if a.state == stateAnswering {
        if len(a.outputs) < len(a.Response.Commands) {
            var styledCmds []string
            for _, cmd := range a.Response.Commands {
                styledCmds = append(styledCmds, fmt.Sprintf("$ %s", a.wrap(cmd)))
            }
            styledCmdStr := strings.Join(styledCmds, "\n")
            builder.WriteString(a.styles.Command.Render(styledCmdStr) + "\n\n")
            builder.WriteString(a.styles.Comment.Render("Press y to execute all commands or esc to exit"))
        } else if len(a.answers) < len(a.Response.Questions) {
            question := a.Response.Questions[len(a.answers)]
            builder.WriteString(a.styles.QuestionTag.String() + " " + a.styles.Question.Render(a.wrap(question)))
            builder.WriteString("\n\n")
            builder.WriteString(a.textarea.View())
            builder.WriteString("\n")
            builder.WriteString(a.styles.Comment.Render(fmt.Sprintf("Question %d of %d: press ctrl+d to submit answer", len(a.answers)+1, len(a.Response.Questions))))
        }
    }
    */
    
    // Use String method to get the final string
    view := builder.String()
    return lipgloss.Place(a.common.Width, a.common.Height, lipgloss.Left, lipgloss.Top, a.styles.App.Render(view))
}
