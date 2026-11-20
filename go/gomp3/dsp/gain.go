package dsp

import "math"

// Limiter is a simple peak limiter with a soft safety clipper.
type Limiter struct {
	enabled      bool
	sampleRate   float64
	env          float64
	attackCoeff  float64
	releaseCoeff float64
	ceiling      float64
}

// NewLimiter creates a limiter for the given sample rate.
func NewLimiter(sampleRate int, enabled bool) *Limiter {
	l := &Limiter{}
	l.Update(enabled, sampleRate)
	return l
}

// Update changes the limiter settings.
func (l *Limiter) Update(enabled bool, sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if l.sampleRate != float64(sampleRate) {
		l.attackCoeff = 1 - math.Exp(-1/(0.001*float64(sampleRate)))
		l.releaseCoeff = 1 - math.Exp(-1/(0.100*float64(sampleRate)))
		l.sampleRate = float64(sampleRate)
	}
	l.enabled = enabled
	l.ceiling = 0.97
}

// Reset clears the gain-reduction envelope.
func (l *Limiter) Reset() { l.env = 0 }

// Process limits a stereo pair. The soft safety clip always runs so MyGo's
// output mixer never has to hard-clip the signal.
func (l *Limiter) Process(lv, rv float64) (float64, float64) {
	if l.enabled {
		peak := math.Max(math.Abs(lv), math.Abs(rv))
		target := 0.0
		if peak > l.ceiling {
			target = 1 - l.ceiling/peak
		}
		coeff := l.releaseCoeff
		if target > l.env {
			coeff = l.attackCoeff
		}
		l.env += (target - l.env) * coeff
		g := math.Max(0, 1-l.env)
		lv *= g
		rv *= g
	}
	return softClip(lv), softClip(rv)
}

// ApplyBalance pans a stereo pair with equal power and applies channel switches.
func ApplyBalance(l, r, balance float64, leftOn, rightOn bool) (float64, float64) {
	balance = Clamp(balance, -1, 1)
	angle := (balance + 1) * math.Pi / 4
	lg := math.Sqrt2 * math.Cos(angle)
	rg := math.Sqrt2 * math.Sin(angle)
	if !leftOn {
		lg = 0
	}
	if !rightOn {
		rg = 0
	}
	return l * lg, r * rg
}

// softClip leaves quiet samples untouched and smoothly bends peaks into ±1.
func softClip(x float64) float64 {
	const threshold = 0.9
	a := math.Abs(x)
	if a <= threshold || math.IsNaN(x) || math.IsInf(x, 0) {
		if math.IsNaN(x) {
			return 0
		}
		if math.IsInf(x, 1) {
			return 1
		}
		if math.IsInf(x, -1) {
			return -1
		}
		return x
	}
	y := threshold + (1-threshold)*math.Tanh((a-threshold)/(1-threshold))
	if x < 0 {
		return -y
	}
	return y
}
