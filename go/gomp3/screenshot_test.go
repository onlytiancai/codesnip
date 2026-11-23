package main

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

func TestScreenshot(t *testing.T) {
	if os.Getenv("RENDER_PNG") == "" {
		t.Skip("set RENDER_PNG=1 to render screenshots")
	}
	names := []string{"basic", "advanced-10", "expert", "surround"}
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	for i, name := range names {
		a := testApp(t)
		a.model.EQTab = i
		if i == 1 {
			a.model.AdvancedMode = 0
			a.model.Pipeline.Fixed10.Enabled = true
			a.model.Pipeline.Fixed10.Gains = [10]float64{4, 3, 1, -2, -4, 2, 5, 3, 1, -1}
		}
		if i == 2 {
			b1 := dsp.NewParametricBand()
			b1.Type = dsp.Lowpass
			b1.Freq = 320
			b2 := dsp.NewParametricBand()
			b2.UseBandwidth = true
			b2.Freq = 2500
			b2.GainDB = 3.5
			a.model.Pipeline.Params = []dsp.ParametricBand{b1, b2}
		}
		img := ui.Render(a.view, 1080, 760, 2)
		path := filepath.Join("docs", "screenshot-"+name+".png")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			f.Close()
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", path)
	}

	a := testApp(t)
	a.model.EQTab = 1
	a.model.AdvancedMode = 1
	img := ui.Render(a.view, 1080, 760, 2)
	path := filepath.Join("docs", "screenshot-advanced-31.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", path)
}
