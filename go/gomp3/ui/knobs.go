package ui

import (
	"fmt"

	"github.com/egoist/mygo/ui"
)

// Knob is a horizontal slider. It keeps MyGo's pointer, keyboard and
// accessibility behavior (SliderBase) while drawing a track and a round thumb.
// The name is kept for call-site compatibility; visually it is a slider.
func Knob(c *ui.Context, m *Model, key, label string, value *float64, lo, hi, step float64, unit string, after func()) {
	ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
		ui.Text(c, label).Width(64).FontSize(12).TextColor(colorText)
		s := ui.SliderBase(c.Key(key), value, lo, hi).
			Step(step).
			Height(24).
			PaddingX(10).
			Grow(1).
			FocusRing(false).
			Label(fmt.Sprintf("%s %s", label, unit)).
			Draw(func(p *ui.Painter, r ui.Rect) {
				pad := float32(10)
				trackY := r.Y + r.H/2 - 2
				trackW := r.W - pad*2
				p.Fill(ui.Rect{X: r.X + pad, Y: trackY, W: trackW, H: 4}, colorTrack, 2)
				ratio := 0.0
				if hi > lo {
					ratio = (*value - lo) / (hi - lo)
				}
				if ratio < 0 {
					ratio = 0
				}
				if ratio > 1 {
					ratio = 1
				}
				cx := r.X + pad + trackW*float32(ratio)
				p.Fill(ui.Rect{X: cx - 9, Y: r.Y + 2, W: 18, H: r.H - 4}, colorGreen, 9)
				p.Fill(ui.Rect{X: cx - 3, Y: r.Y + r.H/2 - 3, W: 6, H: 6}, colorGreenDark, 3)
			})
		if s.Changed() {
			if after != nil {
				after()
			}
			m.MarkDirty()
		}
		ui.Text(c, fmt.Sprintf("%+.1f %s", *value, unit)).Width(70).TextAlign(ui.Center).FontSize(12).Bold().TextColor(colorText)
	})
}
