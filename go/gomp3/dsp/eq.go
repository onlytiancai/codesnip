package dsp

import (
	"fmt"
	"math"
	"strings"
)

type filterDescriptor struct {
	key   string
	coeff Coefficients
}

type stereoBiquad struct {
	key   string
	coeff Coefficients
	l, r  *Biquad
}

// FilterChain cascades all enabled EQ biquads, with independent state per channel.
type FilterChain struct {
	items      []*stereoBiquad
	sampleRate int
	signature  string
}

// NewFilterChain builds a chain for a pipeline and sample rate.
func NewFilterChain(p Pipeline, sampleRate int) *FilterChain {
	c := &FilterChain{}
	c.Update(p, sampleRate)
	return c
}

// Update refreshes coefficients, rebuilding only when the filter set changes.
func (c *FilterChain) Update(p Pipeline, sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	descs := buildDescriptors(p, float64(sampleRate))
	keys := make([]string, len(descs))
	for i, d := range descs {
		keys[i] = d.key
	}
	sig := strings.Join(keys, "\x00")

	if c.sampleRate != sampleRate || sig != c.signature {
		c.items = c.items[:0]
		for _, d := range descs {
			c.items = append(c.items, &stereoBiquad{
				key:   d.key,
				coeff: d.coeff,
				l:     NewBiquad(d.coeff),
				r:     NewBiquad(d.coeff),
			})
		}
		c.sampleRate = sampleRate
		c.signature = sig
		return
	}

	for i, d := range descs {
		if !sameCoefficients(c.items[i].coeff, d.coeff) {
			c.items[i].coeff = d.coeff
			c.items[i].l.SetCoefficients(d.coeff)
			c.items[i].r.SetCoefficients(d.coeff)
		}
	}
}

// ProcessBlock filters interleaved-in-plane stereo samples held in two slices.
func (c *FilterChain) ProcessBlock(l, r []float64) {
	for _, f := range c.items {
		fl, fr := f.l, f.r
		for i := range l {
			l[i] = fl.Process(l[i])
			r[i] = fr.Process(r[i])
		}
	}
}

// MagnitudeDB returns the combined frequency response at freq.
func (c *FilterChain) MagnitudeDB(freq float64) float64 {
	var sum float64
	for _, f := range c.items {
		sum += f.coeff.MagnitudeDB(freq, float64(c.sampleRate))
	}
	return sum
}

// Reset clears every filter's history.
func (c *FilterChain) Reset() {
	for _, f := range c.items {
		f.l.Reset()
		f.r.Reset()
	}
}

// Count reports the number of active biquads.
func (c *FilterChain) Count() int { return len(c.items) }

// Response returns the combined dB response of a pipeline at the given
// frequencies. It is allocation-friendly for UI redraws.
func Response(p Pipeline, sampleRate float64, freqs []float64) []float64 {
	descs := buildDescriptors(p, sampleRate)
	level := 20 * math.Log10(DBToGain(p.FX.PreampDB)*Clamp(p.FX.Boost, 1, 2))
	out := make([]float64, len(freqs))
	for i, f := range freqs {
		sum := level
		for _, d := range descs {
			sum += d.coeff.MagnitudeDB(f, sampleRate)
		}
		out[i] = sum
	}
	return out
}

func buildDescriptors(p Pipeline, sr float64) []filterDescriptor {
	var out []filterDescriptor
	add := func(key string, kind FilterType, freq, gain, q float64) {
		out = append(out, filterDescriptor{
			key:   key,
			coeff: DesignCoefficients(kind, freq, gain, q, sr),
		})
	}

	if p.Basic.Enabled {
		if math.Abs(p.Basic.Bass) > 0.01 {
			add("basic-bass", LowShelf, 100, p.Basic.Bass, 0.707)
		}
		if math.Abs(p.Basic.Mid) > 0.01 {
			add("basic-mid", Peaking, 1000, p.Basic.Mid, 1)
		}
		if math.Abs(p.Basic.Treble) > 0.01 {
			add("basic-treble", HighShelf, 10000, p.Basic.Treble, 0.707)
		}
	}

	if p.Fixed10.Enabled {
		for i, f := range Fixed10Freqs {
			g := p.Fixed10.Gains[i]
			if math.Abs(g) < 0.01 {
				continue
			}
			kind := Peaking
			if i == 0 {
				kind = LowShelf
			} else if i == len(Fixed10Freqs)-1 {
				kind = HighShelf
			}
			add(fmt.Sprintf("fixed10-%d", i), kind, f, g, 1.414)
		}
	}

	if p.Fixed31.Enabled {
		for i, f := range Fixed31Freqs {
			g := p.Fixed31.Gains[i]
			if math.Abs(g) < 0.01 {
				continue
			}
			add(fmt.Sprintf("fixed31-%d", i), Peaking, f, g, 4.3)
		}
	}

	for _, b := range p.Params {
		if !b.Enabled {
			continue
		}
		gainDependent := b.Type == Peaking || b.Type == LowShelf || b.Type == HighShelf
		if gainDependent && math.Abs(b.GainDB) < 0.01 {
			continue
		}
		q := Clamp(b.Q, 0.1, 20)
		if b.UseBandwidth {
			q = BandwidthToQ(b.BandwidthOct, b.Freq, sr)
		}
		add("param-"+b.ID, b.Type, b.Freq, b.GainDB, q)
	}

	return out
}

func sameCoefficients(a, b Coefficients) bool {
	const eps = 1e-9
	return math.Abs(a.B0-b.B0) < eps && math.Abs(a.B1-b.B1) < eps &&
		math.Abs(a.B2-b.B2) < eps && math.Abs(a.A1-b.A1) < eps &&
		math.Abs(a.A2-b.A2) < eps
}
