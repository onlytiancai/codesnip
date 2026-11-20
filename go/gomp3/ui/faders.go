package ui

import (
	"fmt"
	"strconv"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

// GraphicEQ10 draws the classic ten vertical faders.
func GraphicEQ10(c *ui.Context, m *Model) {
	graphicFaders(c, m, dsp.Fixed10Freqs[:], m.Pipeline.Fixed10.Gains[:], "eq10", 58, 152)
}

// GraphicEQ31 draws the third-octave faders inside a horizontal scroller.
func GraphicEQ31(c *ui.Context, m *Model) {
	ui.ScrollHorizontal(c).Height(244).Children(func() {
		graphicFaders(c, m, dsp.Fixed31Freqs[:], m.Pipeline.Fixed31.Gains[:], "eq31", 44, 152)
	})
}

func graphicFaders(c *ui.Context, m *Model, freqs, gains []float64, prefix string, width, height float32) {
	ui.Row(c).Gap(4).AlignItems(ui.Center).Children(func() {
		for i := range gains {
			index := i
			ui.Column(c).Width(width).Gap(3).AlignItems(ui.Center).Children(func() {
				s := ui.SliderBase(c.Key(prefix+"-"+strconv.Itoa(index)), &gains[index], -12, 12).
					Step(0.5).
					Vertical().
					Size(width-8, height).
					PaddingY(10).
					FocusRing(false).
					Label(freqLabel(freqs[index]) + " Hz 增益").
					Draw(func(p *ui.Painter, r ui.Rect) {
						cx := r.X + r.W/2
						top := r.Y + 8
						bottom := r.Y + r.H - 8
						p.Fill(ui.Rect{X: cx - 2, Y: top, W: 4, H: bottom - top}, colorTrack, 2)
						for tick := 0; tick <= 8; tick++ {
							y := top + (bottom-top)*float32(tick)/8
							p.Line(cx-6, y, cx+6, y, 1, colorBorder)
						}
						ratio := (gains[index] + 12) / 24
						if ratio < 0 {
							ratio = 0
						}
						if ratio > 1 {
							ratio = 1
						}
						thumbY := bottom - (bottom-top)*float32(ratio)
						p.Fill(ui.Rect{X: cx - 9, Y: thumbY - 7, W: 18, H: 14}, colorGreen, 4)
						p.Line(cx-5, thumbY, cx+5, thumbY, 1, colorGreenDark)
					})
				if s.Changed() {
					m.MarkDirty()
				}
				ui.Text(c, freqLabel(freqs[index])).FontSize(10).TextColor(colorText)
				ui.Text(c, fmt.Sprintf("%+.1f", gains[index])).FontSize(10).TextColor(colorMuted)
			})
		}
	})
}

func freqLabel(freq float64) string {
	switch {
	case freq >= 1000:
		v := freq / 1000
		if v == float64(int(v)) {
			return strconv.Itoa(int(v)) + "k"
		}
		return strconv.FormatFloat(v, 'g', 3, 64) + "k"
	default:
		if freq == float64(int(freq)) {
			return strconv.Itoa(int(freq))
		}
		return strconv.FormatFloat(freq, 'g', 3, 64)
	}
}
