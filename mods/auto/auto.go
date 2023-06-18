package auto 


import (
    "fmt"

	tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/bubbles/textarea"
    "github.com/charmbracelet/bubbles/cursor"
    "github.com/charmbracelet/mods/common"


)

type Auto struct {
    Response *LLMResponse
    textarea textarea.Model
    styles common.Styles 
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
		"actions": [{"question":"Some question"}]
	}`
	a.Response, _ = a.ParseResponse(testJson)

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

    a.textarea, cmd = a.textarea.Update(msg)
    cmds = append(cmds, cmd)

    switch msg := msg.(type) {
        case tea.KeyMsg:
            if msg.String() == "ctrl+d" {
                a.textarea.Reset()
                //FIXME actually submit an answer
            }
        case cursor.BlinkMsg:
            cmds = append(cmds, textarea.Blink)
    }
    return a, tea.Batch(cmds...) 
}

func (a *Auto) View() string {
    view := ""

    view += a.styles.ContextTag.String() + " " + a.styles.Context.Render(a.Response.Context)
    view += "\n\n"
    view += a.styles.GoalTag.String() + " " + a.styles.Goal.Render(a.Response.Goal)
    view += "\n\n"
    view += a.styles.QuestionTag.String() + " " + a.styles.Question.Render(a.Response.Actions[0].String())
    view += "\n\n"
    view += a.textarea.View()
    view += "\n"
    view += a.styles.Comment.Render("press ctrl+d to submit answer")
    return a.styles.App.Render(view)
}

func (a *Auto) AddPrompt(prompt string) string {
    // FIXME 
    return fmt.Sprintf(InitialPrompt, prompt)
}



const InitialPrompt = `
{
    "prompt": "You are part of a program that helps the user to build complex projects. You are able to carry two type of actions: to ask the user clarifying questions such that you don't need to make any assumptions, and to execute commands on the user's terminal.
    Your response should be structured as JSON and include the fields context, goal and actions. See a description of each field below:

    'context': use this field to summarize in a paragraph the result of the previous actions. In other words, you use this field to summarize your current understanding of the situation.
    'goal': given the current context, you must summarize the desire goal that you want to achieve next.
    'actions': this is a list of the type { 'question': "How are you?" } or { 'cmd': 'tree ./ ' }. As exampled, you can either ask questions to the user or run commands in the terminal. You are not allowed to access files outside of the current directory neither directly with your commands or indirectly, for example, by a program you write and execute. When you actions include writting code, there is a strong preference to use Golang.

    Here is a simple example of your response
    { 'context': 'The user wants to understand how long has the current machine being on',
      'goal': 'I will use the program uptime to get extract such information',
      'actions': [{'cmd': 'uptime'},{'question':'is there a particular format in which you want the time to be shown?'}]
    }
    "

    In the current directory is a git repository of the code that parses, executes your responses and feeds them back to you.
    There is a file called plan.md which describes how to make this tool smarter and more useful. Your goal is to help us perfect this plan and also to carry out the plan.
    %s

    "previous": [],
}
`
