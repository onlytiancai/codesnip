package ui

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

var graphGridFreqs = []float64{20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000}

// ResponseGraph draws the combined EQ curve and the active graphic bands.
func ResponseGraph(c *ui.Context, m *Model) {
	sr := m.SampleRate
	if sr <= 0 {
		sr = 44100
	}
	const points = 180
	freqs := make([]float64, points)
	for i := 0; i < points; i++ {
		t := float64(i) / float64(points-1)
		freqs[i] = 20 * math.Pow(1000, t)
	}
	response := dsp.Response(m.Pipeline, float64(sr), freqs)

	ui.Box(c).Height(156).Background(colorSurface2).Border(1, colorBorder).Radius(8).Draw(func(p *ui.Painter, r ui.Rect) {
		plot := ui.Rect{X: r.X + 38, Y: r.Y + 12, W: r.W - 48, H: r.H - 34}
		p.Fill(plot, ui.Hex("#F9FAFB"), 4)

		for db := -18.0; db <= 18; db += 6 {
			y := dbY(plot, db)
			p.Line(plot.X, y, plot.X+plot.W, y, 1, ui.Hex("#DDE1E5"))
			p.Text(r.X+4, y-6, fmt.Sprintf("%+.0f", db), 9, colorMuted)
		}
		for _, f := range graphGridFreqs {
			x := freqX(plot, f)
			p.Line(x, plot.Y, x, plot.Y+plot.H, 1, ui.Hex("#E4E7EA"))
			p.Text(x-12, plot.Y+plot.H+7, freqLabel(f), 9, colorMuted)
		}
		p.Line(plot.X, dbY(plot, 0), plot.X+plot.W, dbY(plot, 0), 1, colorBorder)

		for i, gain := range bandGains(m) {
			if math.Abs(gain) < 0.01 {
				continue
			}
			f0 := bandFreqs(m)[i]
			coeff := dsp.DesignCoefficients(dsp.Peaking, f0, gain, bandQ(m, i), float64(sr))
			pts := make([][2]float32, points)
			for j, f := range freqs {
				pts[j] = [2]float32{freqX(plot, f), dbY(plot, coeff.MagnitudeDB(f, float64(sr)))}
			}
			p.StrokePolyline(pts, 1.5, bandColors[i%len(bandColors)].Alpha(0.55))
		}

		pts := make([][2]float32, points)
		for i, f := range freqs {
			pts[i] = [2]float32{freqX(plot, f), dbY(plot, response[i])}
		}
		p.StrokePolyline(pts, 2.5, colorGreenDark)
	})
}

func freqX(r ui.Rect, freq float64) float32 {
	t := (math.Log(freq) - math.Log(20)) / math.Log(1000)
	return r.X + r.W*float32(t)
}

func dbY(r ui.Rect, db float64) float32 {
	if db > 18 {
		db = 18
	}
	if db < -18 {
		db = -18
	}
	return r.Y + r.H*float32((18-db)/36)
}

func bandFreqs(m *Model) []float64 {
	if m.AdvancedMode == 1 {
		return dsp.Fixed31Freqs[:]
	}
	return dsp.Fixed10Freqs[:]
}

func bandGains(m *Model) []float64 {
	if m.AdvancedMode == 1 {
		return m.Pipeline.Fixed31.Gains[:]
	}
	return m.Pipeline.Fixed10.Gains[:]
}

func bandQ(m *Model, _ int) float64 {
	if m.AdvancedMode == 1 {
		return 4.3
	}
	return 1.414
}
