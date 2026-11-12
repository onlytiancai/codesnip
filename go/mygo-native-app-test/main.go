package main

import (
	"log"
	"runtime"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

// app is the state the window shows. Its view builds the interface from
// it, on the main thread, whenever the window needs a frame: after input,
// after Window.Update, and while something animates.
type app struct {
	win *mygo.Window

	source  string // 源文本
	result  string // 译文
	from    string // 源语言
	to      string // 目标语言
	loading bool
	errMsg  string

	// doTranslate asks the local model for a translation. Tests replace it.
	doTranslate func(from, to, text string) (string, error)
	// publish changes the state on the main thread, where the view builds
	// the interface.
	publish func(fn func())
}

// newApp returns the state the window starts with.
func newApp() *app {
	a := &app{
		from:        langs[0].label,
		to:          "English",
		doTranslate: translateText,
	}
	a.publish = func(fn func()) {
		if a.win != nil {
			// Window.Update runs fn on the main thread and draws the
			// frame that shows it.
			a.win.Update(fn)
			return
		}
		fn()
	}
	return a
}

// view builds one frame of the interface.
func (a *app) view(c *ui.Context) {
	t := c.Theme()
	// ⌘↩ 翻译，Windows 和 Linux 上是 Ctrl↩。
	c.OnShortcut(ui.Cmd, ui.KeyEnter, a.translate)
	ui.Column(c).Fill().Padding(20).Gap(12).Children(func() {
		ui.Text(c, "本地翻译").FontSize(22).Bold()

		ui.TextArea(c.Key("source"), &a.source).
			Label("源文本").
			Placeholder("输入要翻译的文本…").
			Lines(3, 8)

		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Select(c.Key("from"), &a.from, langLabels).Label("源语言")
			ui.Button(c.Key("swap"), "↔").Tooltip("交换语言").OnClick(a.swap)
			ui.Select(c.Key("to"), &a.to, langLabels).Label("目标语言")
		})

		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.PrimaryButton(c.Key("translate"), "翻译").
				Disabled(a.loading || strings.TrimSpace(a.source) == "").
				OnClick(a.translate)
			if a.loading {
				ui.Spinner(c).Size(18, 18).Label("翻译中")
			}
			ui.Text(c, translateHint()).FontSize(12).TextColor(t.TextMuted)
		})

		if a.errMsg != "" {
			ui.Text(c, a.errMsg).TextColor(t.Danger)
		}

		ui.TextArea(c.Key("result"), &a.result).
			Label("译文").
			ReadOnly(true).
			Placeholder("译文会显示在这里").
			Lines(3, 10)

		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Button(c.Key("copy"), "复制").Disabled(a.result == "").OnClick(func() {
				c.WriteClipboard(a.result)
				c.Toast("已复制")
			})
		})
	})
}

// swap exchanges the source and the target language.
func (a *app) swap() {
	a.from, a.to = a.to, a.from
}

// translateHint names the shortcut that translates, in the words of the
// platform it runs on.
func translateHint() string {
	if runtime.GOOS == "darwin" {
		return "⌘↩ 翻译"
	}
	return "Ctrl+↩ 翻译"
}

// translate asks the local model for a translation of the source text. The
// request takes a moment, so it runs on its own goroutine and publishes
// what it gets back on the main thread.
func (a *app) translate() {
	// Take what the view shows now: the goroutine must not read the
	// state of the interface, which the user can change while it runs.
	text, from, to := a.source, a.from, a.to
	if strings.TrimSpace(text) == "" {
		return
	}
	if from == to {
		a.errMsg = "源语言和目标语言相同"
		return
	}
	a.loading = true
	a.errMsg = ""
	a.result = ""
	go func() {
		out, err := a.doTranslate(from, to, text)
		a.publish(func() {
			a.loading = false
			if err != nil {
				a.errMsg = err.Error()
				return
			}
			a.result = out
		})
	}()
}

func main() {
	a := newApp()
	mygo.App.WhenReady(func() {
		win := mygo.NewWindow(mygo.WindowOptions{
			Title:     "MyGo 翻译",
			Width:     560,
			Height:    520,
			MinWidth:  460,
			MinHeight: 420,
			// Opens where the user left it last time.
			StateKey: "main",
			// The window shows the interface MyGo draws, not a web page.
			Content: ui.View(a.view),
		})
		a.win = win
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
