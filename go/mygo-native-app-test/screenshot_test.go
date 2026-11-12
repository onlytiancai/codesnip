package main

import (
	"image/png"
	"os"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
)

// TestScreenshot draws the interface with a real translation in it and
// writes it to docs/screenshot.png, the picture the README shows. It needs
// the Ollama server, so it only runs when RENDER_PNG is set:
//
//	RENDER_PNG=1 go test -run TestScreenshot
func TestScreenshot(t *testing.T) {
	if os.Getenv("RENDER_PNG") == "" {
		t.Skip("RENDER_PNG is not set")
	}
	a := newApp()
	a.source = "我喜欢人工智能。今天天气很好，我们打算去公园散步，然后回家做晚饭。"
	published := make(chan func(), 1)
	a.publish = func(fn func()) { published <- fn }
	a.translate()
	select {
	case fn := <-published:
		fn()
	case <-time.After(60 * time.Second):
		t.Fatal("模型没有回答")
	}
	img := ui.Render(a.view, 560, 520, 2)
	f, err := os.Create("docs/screenshot.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}
