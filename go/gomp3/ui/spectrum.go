package ui

import (
	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

// SpectrumView draws the live FFT bars with a logarithmic frequency axis.
func SpectrumView(c *ui.Context, spectrum *dsp.Spectrum) {
	ui.Box(c).Height(142).Background(colorSurface2).Border(1, colorBorder).Radius(8).Draw(func(p *ui.Painter, r ui.Rect) {
		plot := ui.Rect{X: r.X + 10, Y: r.Y + 8, W: r.W - 20, H: r.H - 26}
		p.Fill(plot, ui.Hex("#F9FAFB"), 4)

		for _, f := range graphGridFreqs {
			x := freqX(plot, f)
			p.Line(x, plot.Y, x, plot.Y+plot.H, 1, ui.Hex("#E4E7EA"))
			p.Text(x-11, plot.Y+plot.H+5, freqLabel(f), 9, colorMuted)
		}
		for i := 1; i < 4; i++ {
			y := plot.Y + plot.H*float32(i)/4
			p.Line(plot.X, y, plot.X+plot.W, y, 1, ui.Hex("#ECEFF1"))
		}

		if spectrum == nil {
			p.Text(plot.X+12, plot.Y+16, "打开音频后显示频谱", 12, colorMuted)
			return
		}
		bars := spectrum.Snapshot()
		if len(bars) == 0 {
			return
		}
		n := len(bars)
		barW := (plot.W - 4) / float32(n)
		for i, v := range bars {
			h := plot.H * float32(v)
			if h < 1 {
				h = 1
			}
			x := plot.X + 2 + float32(i)*barW
			col := colorGreen
			if v > 0.82 {
				col = colorGreenDark
			}
			p.Fill(ui.Rect{X: x, Y: plot.Y + plot.H - h, W: barW + 0.6, H: h}, col, 1)
		}
	})
}
