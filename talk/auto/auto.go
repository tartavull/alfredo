package auto

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

    "github.com/tartavull/alfredo/talk/common"
    "github.com/tartavull/alfredo/talk/components/tabs"

    "github.com/tartavull/alfredo/talk/components/chat"
    "github.com/tartavull/alfredo/talk/components/history"
)


type Auto struct {
    common *common.Common
    picked pane
    tabs *tabs.Tabs
	panes  []tea.Model
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

func New(c *common.Common) *Auto {
    a := &Auto{
        common: c,
        picked: paneChat,
		tabs:   InitTabs(c),
        panes: []tea.Model{
            chat.New(c),
            history.New(c),
        },
    }
    return a
}

func (a *Auto) Init() tea.Cmd {
    a.panes[0].Init()
    return nil
}

func InitTabs(c *common.Common) *tabs.Tabs {
	ts := make([]string, paneLast)
	for i, b := range []pane{paneChat, paneHistory} {
		ts[i] = b.String()
	}
	t := tabs.New(c, ts)
	return t
}

func (a *Auto) SetSize(width, height int) {
    a.common.SetSize(width, height)
}

func (a *Auto) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmds := make([]tea.Cmd, 0)
	switch msg := msg.(type) {
	case tabs.ActiveTabMsg:
		a.picked = pane(msg)
        //TODO sent pane active and inactive messages
    case tea.WindowSizeMsg:
        a.SetSize(msg.Width, msg.Height)
	}

	_, cmd := a.tabs.Update(msg)
	cmds = append(cmds, cmd)

	_, cmd = a.panes[a.picked].Update(msg)
	cmds = append(cmds, cmd)
	return a, tea.Batch(cmds...)
}

func (a *Auto) View() string {
    var builder strings.Builder
    builder.WriteString(a.common.Styles.Logo.Render(" Alfredo "))
    builder.WriteString("\n\n")
    builder.WriteString(a.tabs.View())
    builder.WriteString("\n\n")
    builder.WriteString(a.panes[a.picked].View())
    return lipgloss.Place(a.common.Width, a.common.Height, 
        lipgloss.Left, lipgloss.Top, 
        a.common.Styles.App.Render(builder.String()))
}
