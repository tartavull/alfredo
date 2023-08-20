package common

import (
	zone "github.com/lrstanley/bubblezone"
	"github.com/muesli/reflow/truncate"
	"github.com/muesli/termenv"
)

// TruncateString is a convenient wrapper around truncate.TruncateString.
func TruncateString(s string, max int) string {
	if max < 0 {
		max = 0
	}
	return truncate.StringWithTail(s, uint(max), "…")
}

type contextKey struct {
	name string
}

// Common is a struct all components should embed.
type Common struct {
	Width, Height int
	Styles        *Styles
	Zone          *zone.Manager
	Output        *termenv.Output
}

// NewCommon returns a new Common struct.
func NewCommon(width, height int,s *Styles) Common {
	return Common{
		Width:   width,
		Height:  height,
        Styles: s,
		Zone:    zone.New(),
	}
}

// SetSize sets the width and height of the common struct.
func (c *Common) SetSize(width, height int) {
	c.Width = width
	c.Height = height
}

func strptr(s string) *string {
	return &s
}
