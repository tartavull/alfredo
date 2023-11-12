package ui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

    "github.com/tartavull/alfredo/talk/common"
    "github.com/tartavull/alfredo/talk/components/tabs"

    "github.com/tartavull/alfredo/talk/components/chat"
    "github.com/tartavull/alfredo/talk/components/history"
)


type UI struct {
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

func New(c *common.Common) *UI {
    a := &UI{
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

func (ui *UI) Init() tea.Cmd {
    ui.panes[0].Init()
    ui.panes[1].Init()
    return nil
}

func InitTabs(c *common.Common) *tabs.Tabs {
	t := tabs.New(c, []string{"Chat", "History"})
	return t
}

func (a *UI) SetSize(width, height int) {
    a.common.SetSize(width, height)
}

func (ui *UI) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	cmds := make([]tea.Cmd, 0)
	switch msg := msg.(type) {
	case tabs.ActiveTabMsg:
		ui.picked = pane(msg)
        //TODO sent pane active and inactive messages
    case tea.WindowSizeMsg:
        ui.SetSize(msg.Width, msg.Height)
    case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			return ui, tea.Quit
		}
	}

	_, cmd := ui.tabs.Update(msg)
	cmds = append(cmds, cmd)

	_, cmd = ui.panes[ui.picked].Update(msg)
	cmds = append(cmds, cmd)
	return ui, tea.Batch(cmds...)
}

func (ui *UI) View() string {
    var builder strings.Builder
    builder.WriteString(ui.common.Styles.Logo.Render(" Alfredo "))
    builder.WriteString("\n\n")
    builder.WriteString(ui.tabs.View())
    builder.WriteString("\n\n")
    builder.WriteString(ui.panes[ui.picked].View())
    return lipgloss.Place(ui.common.Width, ui.common.Height, 
        lipgloss.Left, lipgloss.Top, 
        ui.common.Styles.App.Render(builder.String()))
}
