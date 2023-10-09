package history

import (
	"strings"
	tea "github.com/charmbracelet/bubbletea"

    "github.com/tartavull/alfredo/talk/common"
)
    
type History struct {
    common *common.Common
}

func New(c *common.Common) *History {
    return &History{
        common: c, 
    }
}

func (h *History) Init() tea.Cmd {
    return nil
}

func (h *History) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    return h, nil
}

func (h *History) View() string {
    var builder strings.Builder
    return builder.String()
}
