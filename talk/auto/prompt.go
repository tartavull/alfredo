package auto

import (
    "log"
    _ "io/ioutil"

	"encoding/json"
	"github.com/tartavull/alfredo/talk/sandbox"
)

func (a *Auto) buildPrompt() string {
    r := Request{
        Prompt: "",
        Feedback: "",
        Answers: make([]Answer, len(a.Response.Questions)),
        Outputs: make([]Output, len(a.Response.Commands)),
        Previous: make([]string, 2),
    }
    for i, question := range a.Response.Questions{
        r.Answers[i] = Answer{ Question: question, Answer: a.answers[i]}
	}
    for i, cmd := range a.Response.Commands{
        r.Outputs[i] = Output{ Command:cmd, Output: a.outputs[i]}
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
