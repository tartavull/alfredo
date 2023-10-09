package chat

import (
    "log"
    "os"
	"strings"
	"encoding/json"
    "fmt"

	tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/bubbles/cursor"
	"github.com/mitchellh/go-wordwrap"

    "github.com/tartavull/alfredo/talk/common"
	"github.com/charmbracelet/bubbles/viewport"
    "github.com/charmbracelet/bubbles/textarea"
    "github.com/tartavull/alfredo/talk/sandbox"
)

type State int

const (
    stateAnswering State = iota
    stateCompleted
)
    
type Chat struct {
    common *common.Common

    textarea textarea.Model
    state State
    Response *LLMResponse
    answers []string
    outputs []sandbox.Result
    viewport   viewport.Model
}


func New(co *common.Common) *Chat {
    c := &Chat{
        common: co,
        textarea: textarea.New(),
        viewport: viewport.New(0, 0),
    }
    c.viewport.YPosition = 0
	c.textarea.Placeholder = ">"
    c.textarea.ShowLineNumbers = false
    c.textarea.Focus()

    data, err := os.ReadFile("chat/llm_1.json")
    if err != nil {
        log.Fatal(err)
    }
	c.Response, _ = ParseResponse(string(data))
    c.state = stateAnswering
    c.answers = []string{}

    return c
}


func (c *Chat) Init() tea.Cmd {
    return nil
}


func (c *Chat) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    var cmd tea.Cmd
    cmds := make([]tea.Cmd, 0)

    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        c.SetSize(msg.Width, msg.Height)
    }

	if c.state == stateAnswering {
        switch msg := msg.(type) {
        case tea.KeyMsg:
			if len(c.outputs) < len(c.Response.Commands) {
				if msg.String() == "y" {
					for _, cmd := range c.Response.Commands {
						c.outputs = append(c.outputs, sandbox.ExecuteCommand(cmd))
					}
				}
			} else if len(c.answers) < len(c.Response.Questions) {
				c.textarea, cmd = c.textarea.Update(msg)
				cmds = append(cmds, cmd)
				if msg.String() == "ctrl+d" {
					c.answers = append(c.answers, c.textarea.Value())
					c.textarea.Reset()
				}
            } else {
				c.state = stateCompleted
            }
        case cursor.BlinkMsg:
        	cmds = append(cmds, textarea.Blink)
        }
    }
    if c.state == stateCompleted {
        //prompt := buildPrompt()
    }
    return c, tea.Batch(cmds...)
}


func (c *Chat) View() string {
    var builder strings.Builder
    builder.WriteString(c.common.Styles.ContextTag.String() + " " + c.common.Styles.Context.Render(c.wrap(c.Response.Context)))
    builder.WriteString("\n\n")
    builder.WriteString(c.common.Styles.GoalTag.String() + " " + c.common.Styles.Goal.Render(c.wrap(c.Response.Goal)))
    builder.WriteString("\n\n")
    
    if c.state == stateAnswering {
        if len(c.outputs) < len(c.Response.Commands) {
            var styledCmds []string
            for _, cmd := range c.Response.Commands {
                styledCmds = append(styledCmds, fmt.Sprintf("$ %s", c.wrap(cmd)))
            }
            styledCmdStr := strings.Join(styledCmds, "\n")
            builder.WriteString(c.common.Styles.Command.Render(styledCmdStr) + "\n\n")
            builder.WriteString(c.common.Styles.Comment.Render("Press y to execute all commands or esc to exit"))
        } else if len(c.answers) < len(c.Response.Questions) {
            question := c.Response.Questions[len(c.answers)]
            builder.WriteString(c.common.Styles.QuestionTag.String() + " " + c.common.Styles.Question.Render(c.wrap(question)))
            builder.WriteString("\n\n")
            builder.WriteString(c.textarea.View())
            builder.WriteString("\n")
            builder.WriteString(c.common.Styles.Comment.Render(fmt.Sprintf("Question %d of %d: press ctrl+d to submit answer", len(c.answers)+1, len(c.Response.Questions))))
        }
    }
    return builder.String()
}

func (c *Chat) Focus() tea.Cmd {
    return textarea.Blink
}

func (c *Chat) SetSize(width, height int) {
    c.textarea.SetWidth(width)
    c.textarea.MaxHeight = height / 2
    c.viewport.Width = width
    c.viewport.Height = height
}

func (c *Chat) wrap(in string) string {
    return wordwrap.WrapString(in, uint(c.textarea.Width()/2))
}

func (c *Chat) buildPrompt() string {
    r := Request{
        Prompt: "",
        Feedback: "",
        Answers: make([]Answer, len(c.Response.Questions)),
        Outputs: make([]Output, len(c.Response.Commands)),
        Previous: make([]string, 2),
    }
    for i, question := range c.Response.Questions{
        r.Answers[i] = Answer{ Question: question, Answer: c.answers[i]}
	}
    for i, cmd := range c.Response.Commands{
        r.Outputs[i] = Output{ Command:cmd, Output: c.outputs[i]}
    }

    /*
    data, err := ioutil.ReadFile("chat/llm_0.json")
    if err != nil {
        log.Fatal(err)
    }
    r.Previous[0] = string(data)

    data, err = ioutil.ReadFile("chat/user_0.json")
    if err != nil {
        log.Fatal(err)
    }
    r.Previous[1] = string(data)
    */

    b, err := json.Marshal(r)
    if err != nil {
        log.Fatal(err)
    }
	return string(b)
}

type Request struct {
	Prompt   string           `json:"prompt"`
	Feedback string           `json:"feedback"`
	Answers  []Answer         `json:"answers"`
	Outputs  []Output         `json:"outputs"`
	Previous []string         `json:"previous"`
}

type Output struct {
    Command  string         `json:"command"`
    Output   sandbox.Result `json:"output"`
}

type Answer struct {
    Question string  `json:"question"`
    Answer string `json:"answer"`
}
