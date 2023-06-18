package main

import (
	"math/rand"
	"regexp"

    "github.com/charmbracelet/mods/common"
)

var examples = map[string]string{
	"Write new sections for a readme": `cat README.md | mods "write a new section to this README documenting a pdf sharing feature"`,
	"Editorialze your video files":    `ls ~/vids | mods -f "summarize each of these titles, group them by decade" | glow`,
	"Let GPT pick something to watch": `ls ~/vids | mods "Pick 5 action packed shows from the 80s from this list" | gum choose | xargs vlc`,
}

func randomExample() (string, string) {
	keys := make([]string, 0, len(examples))
	for k := range examples {
		keys = append(keys, k)
	}
	desc := keys[rand.Intn(len(keys))] //nolint:gosec
	return desc, examples[desc]
}

func cheapHighlighting(s common.Styles, code string) string {
	code = regexp.
		MustCompile(`"([^"\\]|\\.)*"`).
		ReplaceAllStringFunc(code, func(x string) string {
			return s.Quote.Render(x)
		})
	code = regexp.
		MustCompile(`\|`).
		ReplaceAllStringFunc(code, func(x string) string {
			return s.Pipe.Render(x)
		})
	return code
}
