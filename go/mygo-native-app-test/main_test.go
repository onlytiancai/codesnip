package main

import (
	"errors"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
)

// tester returns a tester for the app, and the channel that carries the
// change the app publishes after the model answered.
func tester(a *app) (*ui.Tester, chan func()) {
	published := make(chan func(), 1)
	a.publish = func(fn func()) { published <- fn }
	return ui.NewTester(a.view, 560, 720), published
}

// publish runs what the app published, on the goroutine of the test, as
// the app runs it on the main thread.
func publish(t *testing.T, published chan func()) {
	t.Helper()
	select {
	case fn := <-published:
		fn()
	case <-time.After(10 * time.Second):
		t.Fatal("模型没有回答")
	}
}

// TestViewTranslate types a text, asks for a translation and copies what
// came back.
func TestViewTranslate(t *testing.T) {
	a := newApp()
	var gotFrom, gotTo, gotText string
	a.doTranslate = func(from, to, text string) (string, error) {
		gotFrom, gotTo, gotText = from, to, text
		return "I like artificial intelligence.", nil
	}
	tt, published := tester(a)

	if err := tt.Click("源文本"); err != nil {
		t.Fatal(err)
	}
	tt.Type("我喜欢人工智能")
	if err := tt.Click("翻译"); err != nil {
		t.Fatal(err)
	}
	publish(t, published)
	tt.Frame()

	if gotText != "我喜欢人工智能" {
		t.Errorf("翻译的文本 %q", gotText)
	}
	if gotFrom != "中文" || gotTo != "English" {
		t.Errorf("语言 %q → %q", gotFrom, gotTo)
	}
	if a.result != "I like artificial intelligence." {
		t.Errorf("译文 %q", a.result)
	}
	if a.loading {
		t.Error("翻译完了还在加载")
	}
	if err := tt.Click("复制"); err != nil {
		t.Fatal(err)
	}
	if got := tt.Clipboard(); got != "I like artificial intelligence." {
		t.Errorf("剪贴板 %q", got)
	}
}

// TestViewTranslateError shows the failure the model reported.
func TestViewTranslateError(t *testing.T) {
	a := newApp()
	a.doTranslate = func(from, to, text string) (string, error) {
		return "", errors.New("无法连接 Ollama（http://localhost:11434），请确认 ollama serve 已启动")
	}
	tt, published := tester(a)

	if err := tt.Click("源文本"); err != nil {
		t.Fatal(err)
	}
	tt.Type("你好")
	if err := tt.Click("翻译"); err != nil {
		t.Fatal(err)
	}
	publish(t, published)
	tt.Frame()

	if !tt.HasText("无法连接 Ollama") {
		t.Errorf("界面上的文字 %q", tt.Texts())
	}
	if a.result != "" {
		t.Errorf("译文 %q", a.result)
	}
	if a.loading {
		t.Error("翻译完了还在加载")
	}
}

// TestViewSameLanguage asks for a translation into the same language,
// which the app refuses to ask the model for.
func TestViewSameLanguage(t *testing.T) {
	a := newApp()
	a.to = "中文"
	asked := false
	a.doTranslate = func(from, to, text string) (string, error) {
		asked = true
		return "不该发生", nil
	}
	tt, _ := tester(a)

	if err := tt.Click("源文本"); err != nil {
		t.Fatal(err)
	}
	tt.Type("你好")
	if err := tt.Click("翻译"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()

	if asked {
		t.Error("语言相同时不该问模型")
	}
	if !tt.HasText("源语言和目标语言相同") {
		t.Errorf("界面上的文字 %q", tt.Texts())
	}
}

// TestViewSwap exchanges the two languages.
func TestViewSwap(t *testing.T) {
	a := newApp()
	tt, _ := tester(a)

	if a.from != "中文" || a.to != "English" {
		t.Fatalf("语言 %q → %q", a.from, a.to)
	}
	if err := tt.Click("↔"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()

	if a.from != "English" || a.to != "中文" {
		t.Errorf("交换后 %q → %q", a.from, a.to)
	}
}

// TestViewEmptySource asks for a translation of nothing, which the app
// does not ask the model for.
func TestViewEmptySource(t *testing.T) {
	a := newApp()
	asked := false
	a.doTranslate = func(from, to, text string) (string, error) {
		asked = true
		return "", nil
	}
	tt, _ := tester(a)

	if err := tt.Click("翻译"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()

	if asked {
		t.Error("没有文本时不该问模型")
	}
}

// TestViewShortcut translates with the keyboard: ⌘↩ on macOS, Ctrl↩
// elsewhere.
func TestViewShortcut(t *testing.T) {
	a := newApp()
	asked := false
	a.doTranslate = func(from, to, text string) (string, error) {
		asked = true
		return "ok", nil
	}
	tt, published := tester(a)

	if err := tt.Click("源文本"); err != nil {
		t.Fatal(err)
	}
	tt.Type("你好")
	tt.Key(ui.Cmd, ui.KeyEnter)
	publish(t, published)
	tt.Frame()

	if !asked {
		t.Error("⌘↩ 该问模型")
	}
}

// TestViewSwapWhileTranslating changes the languages while the model is
// still working. The request keeps the languages it started with, and the
// goroutine that asks the model reads no state of the interface.
func TestViewSwapWhileTranslating(t *testing.T) {
	a := newApp()
	release := make(chan struct{})
	var gotFrom, gotTo string
	a.doTranslate = func(from, to, text string) (string, error) {
		gotFrom, gotTo = from, to
		<-release
		return "ok", nil
	}
	tt, published := tester(a)

	if err := tt.Click("源文本"); err != nil {
		t.Fatal(err)
	}
	tt.Type("你好")
	if err := tt.Click("翻译"); err != nil {
		t.Fatal(err)
	}
	// The user changes the languages while the model is working. The
	// request runs on its own goroutine, so this races with it unless the
	// request keeps what it started with.
	if err := tt.Click("↔"); err != nil {
		t.Fatal(err)
	}
	close(release)
	publish(t, published)
	tt.Frame()

	if gotFrom != "中文" || gotTo != "English" {
		t.Errorf("语言 %q → %q", gotFrom, gotTo)
	}
	if a.from != "English" || a.to != "中文" {
		t.Errorf("交换后 %q → %q", a.from, a.to)
	}
}
