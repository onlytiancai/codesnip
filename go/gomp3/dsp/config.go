package dsp

import (
	"fmt"
	"math"
	"sync/atomic"
)

// FilterType identifies a biquad filter shape.
type FilterType uint8

const (
	Peaking FilterType = iota
	LowShelf
	HighShelf
	Lowpass
	Highpass
	Bandpass
	Notch
)

var filterTypeNames = []string{"峰化", "低频架", "高频架", "低通", "高通", "带通", "陷波"}

func (t FilterType) String() string {
	if int(t) < len(filterTypeNames) {
		return filterTypeNames[t]
	}
	return "未知"
}

// FilterTypes returns the labels shown in the expert EQ filter-type selector.
func FilterTypes() []string {
	out := make([]string, len(filterTypeNames))
	copy(out, filterTypeNames)
	return out
}

// FilterTypeFromLabel reverses String for UI selections.
func FilterTypeFromLabel(label string) FilterType {
	for i, name := range filterTypeNames {
		if name == label {
			return FilterType(i)
		}
	}
	return Peaking
}

// Center frequencies of the fixed graphic EQs.
var (
	Fixed10Freqs = [10]float64{32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000}
	Fixed31Freqs = [31]float64{
		20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
		800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
		12500, 16000, 20000,
	}
)

// BasicEQ is the three-knob tone control.
type BasicEQ struct {
	Enabled           bool
	Bass, Mid, Treble float64
}

// Fixed10EQ is the classic ten-band graphic EQ.
type Fixed10EQ struct {
	Enabled bool
	Gains   [10]float64
}

// Fixed31EQ is the third-octave graphic EQ.
type Fixed31EQ struct {
	Enabled bool
	Gains   [31]float64
}

// ParametricBand is one user-controlled filter in the expert page.
type ParametricBand struct {
	ID           string
	Enabled      bool
	Type         FilterType
	Freq         float64
	GainDB       float64
	Q            float64
	BandwidthOct float64
	UseBandwidth bool
}

func (b ParametricBand) EffectiveQ() float64 {
	if b.UseBandwidth {
		return BandwidthToQ(b.BandwidthOct, b.Freq, 48000)
	}
	return b.Q
}

// Surround controls the virtual headphone soundstage.
type Surround struct {
	Enabled  bool
	LowFreq  float64 // 0..1
	Strength float64 // 0..1
}

// Effects holds gain, balance and protection settings.
type Effects struct {
	PreampDB float64 // -12..+12
	Boost    float64 // 1..2, 1 means 100%
	Balance  float64 // -1..1
	Limiter  bool
	LeftOn   bool
	RightOn  bool
}

// Pipeline is an immutable DSP snapshot shared with the audio goroutine.
type Pipeline struct {
	Basic    BasicEQ
	Fixed10  Fixed10EQ
	Fixed31  Fixed31EQ
	Params   []ParametricBand
	Surround Surround
	FX       Effects
}

// Clone returns a deep copy so the UI can keep editing its own state.
func (p Pipeline) Clone() Pipeline {
	p.Params = append([]ParametricBand(nil), p.Params...)
	return p
}

// DefaultPipeline returns a neutral but usable starting point.
func DefaultPipeline() Pipeline {
	return Pipeline{
		Basic:    BasicEQ{Enabled: true},
		Fixed10:  Fixed10EQ{Enabled: false},
		Fixed31:  Fixed31EQ{Enabled: false},
		Params:   []ParametricBand{},
		Surround: Surround{Enabled: false, LowFreq: 0, Strength: 0.25},
		FX: Effects{
			PreampDB: 0,
			Boost:    1,
			Balance:  0,
			Limiter:  true,
			LeftOn:   true,
			RightOn:  true,
		},
	}
}

var bandID atomic.Uint64

// NewBandID returns a stable ID for a newly added parametric band.
func NewBandID() string {
	return fmt.Sprintf("band-%d", bandID.Add(1))
}

// NewParametricBand creates an enabled peaking band with neutral settings.
func NewParametricBand() ParametricBand {
	return ParametricBand{
		ID:           NewBandID(),
		Enabled:      true,
		Type:         Peaking,
		Freq:         1000,
		GainDB:       0,
		Q:            1,
		BandwidthOct: 1,
		UseBandwidth: false,
	}
}

// DBToGain converts decibels to a linear amplitude gain.
func DBToGain(db float64) float64 {
	return math.Pow(10, db/20)
}

// Clamp keeps v inside [lo, hi].
func Clamp(v, lo, hi float64) float64 {
	if math.IsNaN(v) {
		return lo
	}
	return math.Min(hi, math.Max(lo, v))
}

// BandwidthToQ converts an octave bandwidth to Q using the RBJ relation.
func BandwidthToQ(bw, freq, sampleRate float64) float64 {
	bw = Clamp(bw, 0.1, 4)
	w0 := 2 * math.Pi * Clamp(freq, 20, sampleRate/2*0.99) / sampleRate
	den := 2 * math.Sinh(math.Ln2/2*bw*w0/math.Sin(w0))
	if math.Abs(den) < 1e-9 {
		return 0.1
	}
	return Clamp(1/den, 0.1, 20)
}

// QToBandwidth converts Q back to an octave bandwidth.
func QToBandwidth(q, freq, sampleRate float64) float64 {
	q = Clamp(q, 0.1, 20)
	w0 := 2 * math.Pi * Clamp(freq, 20, sampleRate/2*0.99) / sampleRate
	bw := 2 / math.Ln2 * math.Asinh(1/(2*q)) * math.Sin(w0) / w0
	return Clamp(bw, 0.1, 4)
}
