package ui

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"
)

// Knob is a circular control built on SliderBase so it keeps pointer,
// keyboard and accessibility behavior.
func Knob(c *ui.Context, m *Model, key, label string, value *float64, lo, hi, step float64, unit string, after func()) {
	ui.Column(c).Gap(2).AlignItems(ui.Center).Children(func() {
		s := ui.SliderBase(c.Key(key), value, lo, hi).
			Step(step).
			Vertical().
			Size(74, 74).
			Padding(13).
			FocusRing(false).
			Label(fmt.Sprintf("%s %s", label, unit)).
			Draw(func(p *ui.Painter, r ui.Rect) {
				cx, cy := r.X+r.W/2, r.Y+r.H/2
				radius := float32(math.Min(float64(r.W), float64(r.H)) / 2)
				p.Fill(ui.Rect{X: r.X + 3, Y: r.Y + 3, W: r.W - 6, H: r.H - 6}, colorTrack, radius)
				p.Fill(ui.Rect{X: r.X + 8, Y: r.Y + 8, W: r.W - 16, H: r.H - 16}, colorSurface2, radius-5)

				ratio := (*value - lo) / (hi - lo)
				ratio = math.Max(0, math.Min(1, ratio))
				angle := float32(-135 + ratio*270)
				p.Transform(ui.Rotation(angle), cx, cy, func() {
					p.Line(cx, cy-8, cx, cy-float32(radius-7), 3, colorGreenDark)
				})
				p.Fill(ui.Rect{X: cx - 3, Y: cy - 3, W: 6, H: 6}, colorGreenDark, 3)
			})
		if s.Changed() {
			if after != nil {
				after()
			}
			m.MarkDirty()
		}
		ui.Text(c, label).FontSize(12).TextColor(colorMuted)
		ui.Text(c, fmt.Sprintf("%+.1f %s", *value, unit)).FontSize(12).Bold().TextColor(colorText)
	})
}
