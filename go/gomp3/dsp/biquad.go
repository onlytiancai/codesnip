package dsp

import "math"

// Coefficients holds normalized biquad coefficients (a0 is divided out).
type Coefficients struct {
	B0, B1, B2 float64
	A1, A2     float64
}

// Biquad is a stateful transposed-direct-form-II biquad.
type Biquad struct {
	c      Coefficients
	z1, z2 float64
}

// NewBiquad creates a biquad with the given coefficients.
func NewBiquad(c Coefficients) *Biquad {
	return &Biquad{c: c}
}

// SetCoefficients updates the filter while keeping its state.
func (b *Biquad) SetCoefficients(c Coefficients) {
	b.c = c
}

// Reset clears the filter history.
func (b *Biquad) Reset() {
	b.z1, b.z2 = 0, 0
}

// Process filters one sample.
func (b *Biquad) Process(x float64) float64 {
	y := b.c.B0*x + b.z1
	b.z1 = b.c.B1*x - b.c.A1*y + b.z2
	b.z2 = b.c.B2*x - b.c.A2*y
	// Avoid denormals lingering in the state.
	if math.Abs(b.z1) < 1e-20 {
		b.z1 = 0
	}
	if math.Abs(b.z2) < 1e-20 {
		b.z2 = 0
	}
	return y
}

// MagnitudeDB returns the filter's gain in dB at freq for a given sample rate.
func (b Coefficients) MagnitudeDB(freq, sampleRate float64) float64 {
	w := 2 * math.Pi * Clamp(freq, 1, sampleRate/2*0.999) / sampleRate
	cw, sw := math.Cos(w), math.Sin(w)
	c2, s2 := math.Cos(2*w), math.Sin(2*w)

	numR := b.B0 + b.B1*cw + b.B2*c2
	numI := -(b.B1*sw + b.B2*s2)
	denR := 1 + b.A1*cw + b.A2*c2
	denI := -(b.A1*sw + b.A2*s2)

	num := numR*numR + numI*numI
	den := denR*denR + denI*denI
	if den < 1e-20 {
		return -120
	}
	g := math.Sqrt(num / den)
	if g < 1e-6 {
		return -120
	}
	return 20 * math.Log10(g)
}

// DesignCoefficients returns RBJ cookbook coefficients for a filter shape.
func DesignCoefficients(kind FilterType, freq, gainDB, q, sampleRate float64) Coefficients {
	sr := Clamp(sampleRate, 8000, 192000)
	f0 := Clamp(freq, 20, sr/2*0.98)
	q = Clamp(q, 0.1, 20)
	w0 := 2 * math.Pi * f0 / sr
	cw, sw := math.Cos(w0), math.Sin(w0)
	alpha := sw / (2 * q)
	A := math.Pow(10, gainDB/40)

	switch kind {
	case LowShelf:
		beta := 2 * math.Sqrt(A) * alpha
		b0 := A * ((A + 1) - (A-1)*cw + beta)
		b1 := 2 * A * ((A - 1) - (A+1)*cw)
		b2 := A * ((A + 1) - (A-1)*cw - beta)
		a0 := (A + 1) + (A-1)*cw + beta
		a1 := -2 * ((A - 1) + (A+1)*cw)
		a2 := (A + 1) + (A-1)*cw - beta
		return normalize(b0, b1, b2, a0, a1, a2)
	case HighShelf:
		beta := 2 * math.Sqrt(A) * alpha
		b0 := A * ((A + 1) + (A-1)*cw + beta)
		b1 := -2 * A * ((A - 1) + (A+1)*cw)
		b2 := A * ((A + 1) + (A-1)*cw - beta)
		a0 := (A + 1) - (A-1)*cw + beta
		a1 := 2 * ((A - 1) - (A+1)*cw)
		a2 := (A + 1) - (A-1)*cw - beta
		return normalize(b0, b1, b2, a0, a1, a2)
	case Lowpass:
		b0 := (1 - cw) / 2
		b1 := 1 - cw
		b2 := b0
		a0 := 1 + alpha
		a1 := -2 * cw
		a2 := 1 - alpha
		return normalize(b0, b1, b2, a0, a1, a2)
	case Highpass:
		b0 := (1 + cw) / 2
		b1 := -(1 + cw)
		b2 := b0
		a0 := 1 + alpha
		a1 := -2 * cw
		a2 := 1 - alpha
		return normalize(b0, b1, b2, a0, a1, a2)
	case Bandpass:
		b0 := alpha
		b1 := 0.0
		b2 := -alpha
		a0 := 1 + alpha
		a1 := -2 * cw
		a2 := 1 - alpha
		return normalize(b0, b1, b2, a0, a1, a2)
	case Notch:
		b0 := 1.0
		b1 := -2 * cw
		b2 := 1.0
		a0 := 1 + alpha
		a1 := -2 * cw
		a2 := 1 - alpha
		return normalize(b0, b1, b2, a0, a1, a2)
	case Peaking:
		fallthrough
	default:
		b0 := 1 + alpha*A
		b1 := -2 * cw
		b2 := 1 - alpha*A
		a0 := 1 + alpha/A
		a1 := -2 * cw
		a2 := 1 - alpha/A
		return normalize(b0, b1, b2, a0, a1, a2)
	}
}

func normalize(b0, b1, b2, a0, a1, a2 float64) Coefficients {
	if math.Abs(a0) < 1e-12 {
		return Coefficients{B0: 1}
	}
	return Coefficients{
		B0: b0 / a0,
		B1: b1 / a0,
		B2: b2 / a0,
		A1: a1 / a0,
		A2: a2 / a0,
	}
}
