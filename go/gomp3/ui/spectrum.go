package ui

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

// analyzerDBMin/Max are the vertical range of the combined analyzer. It
// covers the full range of the graphic EQ plus the spectrum on one axis.
const (
	analyzerDBMin = -48.0
	analyzerDBMax = 18.0
)

// AnalyzerView draws the live spectrum as a smooth curve on the same plot as
// the equalizer: each active band in its own colour, the combined response as
// a bold line and the real-time waveform on top.
func AnalyzerView(c *ui.Context, m *Model) {
	sr := m.SampleRate
	if sr <= 0 {
		sr = 44100
	}

	ui.Box(c).Height(190).Background(colorSurface2).Border(1, colorBorder).Radius(8).Draw(func(p *ui.Painter, r ui.Rect) {
		plot := ui.Rect{X: r.X + 40, Y: r.Y + 10, W: r.W - 52, H: r.H - 32}
		p.Fill(plot, ui.Hex("#F9FAFB"), 4)

		// Horizontal dB grid + labels on the left.
		for db := analyzerDBMax; db >= analyzerDBMin; db -= 12 {
			y := analyzerY(plot, db)
			p.Line(plot.X, y, plot.X+plot.W, y, 1, ui.Hex("#E4E7EA"))
			p.Text(r.X+6, y-6, fmt.Sprintf("%+.0f", db), 9, colorMuted)
		}
		// Vertical log-frequency grid + labels along the bottom.
		for _, f := range graphGridFreqs {
			x := freqX(plot, f)
			p.Line(x, plot.Y, x, plot.Y+plot.H, 1, ui.Hex("#E7EAED"))
			p.Text(x-11, plot.Y+plot.H+6, freqLabel(f), 9, colorMuted)
		}
		// The 0 dB reference line.
		p.Line(plot.X, analyzerY(plot, 0), plot.X+plot.W, analyzerY(plot, 0), 1, colorBorder)

		const points = 200
		freqs := make([]float64, points)
		for i := 0; i < points; i++ {
			t := float64(i) / float64(points-1)
			freqs[i] = 20 * math.Pow(1000, t)
		}

		// One coloured curve per active graphic-EQ band, like the reference.
		colors := bandColors
		gains, centerFreqs := analyzerBands(m)
		for i, gain := range gains {
			if math.Abs(gain) < 0.05 || i >= len(centerFreqs) {
				continue
			}
			q := analyzerBandQ(m, i)
			coeff := dsp.DesignCoefficients(dsp.Peaking, centerFreqs[i], gain, q, float64(sr))
			pts := make([][2]float32, points)
			for j, f := range freqs {
				pts[j] = [2]float32{freqX(plot, f), analyzerY(plot, coeff.MagnitudeDB(f, float64(sr)))}
			}
			p.StrokePolyline(pts, 3, colors[i%len(colors)].Alpha(0.45))
		}

		// Combined EQ response: a bold dark curve.
		response := dsp.Response(m.Pipeline, float64(sr), freqs)
		respPts := make([][2]float32, points)
		for i, f := range freqs {
			respPts[i] = [2]float32{freqX(plot, f), analyzerY(plot, response[i])}
		}
		p.StrokePolyline(respPts, 2.4, colorText)

		// Live spectrum as a smooth curve over the same frequency axis.
		bars := m.Spectrum.Snapshot()
		if len(bars) > 1 {
			specPts := make([][2]float32, len(bars))
			for i, v := range bars {
				t := float64(i) / float64(len(bars)-1)
				f := 20 * math.Pow(1000, t)
				db := analyzerDBMin + float64(v)*(analyzerDBMax-analyzerDBMin)*0.92
				specPts[i] = [2]float32{freqX(plot, f), analyzerY(plot, db)}
			}
			p.StrokePolyline(specPts, 1.6, colorGreen.Alpha(0.9))
		}

		// Legend.
		lx := plot.X + plot.W - 168
		p.Fill(ui.Rect{X: lx, Y: plot.Y + 6, W: 10, H: 3}, colorText, 1)
		p.Text(lx+15, plot.Y+2, "合成响应", 9, colorMuted)
		p.Fill(ui.Rect{X: lx + 80, Y: plot.Y + 6, W: 10, H: 3}, colorGreen, 1)
		p.Text(lx+95, plot.Y+2, "实时波形", 9, colorMuted)
	})
}

// analyzerY maps a dB value into the plot rectangle.
func analyzerY(r ui.Rect, db float64) float32 {
	if db > analyzerDBMax {
		db = analyzerDBMax
	}
	if db < analyzerDBMin {
		db = analyzerDBMin
	}
	return r.Y + r.H*float32((analyzerDBMax-db)/(analyzerDBMax-analyzerDBMin))
}

func analyzerBands(m *Model) ([]float64, []float64) {
	if m.AdvancedMode == 1 {
		g := make([]float64, len(m.Pipeline.Fixed31.Gains))
		copy(g, m.Pipeline.Fixed31.Gains[:])
		return g, dsp.Fixed31Freqs[:]
	}
	g := make([]float64, len(m.Pipeline.Fixed10.Gains))
	copy(g, m.Pipeline.Fixed10.Gains[:])
	return g, dsp.Fixed10Freqs[:]
}

func analyzerBandQ(m *Model, _ int) float64 {
	if m.AdvancedMode == 1 {
		return 4.3
	}
	return 1.414
}
