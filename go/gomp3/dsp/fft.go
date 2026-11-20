package dsp

import (
	"math"
	"sync/atomic"
)

// SpectrumBins is the number of log-spaced bars shown in the UI.
const SpectrumBins = 256

const spectrumFFTSize = 2048

// Spectrum computes a smoothed log-frequency spectrum from PCM samples.
type Spectrum struct {
	sampleRate int
	hann       []float64
	re         []float64
	im         []float64
	left       []float64
	right      []float64
	pos        int
	display    atomic.Pointer[[]float32]
}

// NewSpectrum creates an analyzer for a sample rate.
func NewSpectrum(sampleRate int) *Spectrum {
	s := &Spectrum{
		hann:  make([]float64, spectrumFFTSize),
		re:    make([]float64, spectrumFFTSize),
		im:    make([]float64, spectrumFFTSize),
		left:  make([]float64, spectrumFFTSize),
		right: make([]float64, spectrumFFTSize),
	}
	for i := 0; i < spectrumFFTSize; i++ {
		s.hann[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(spectrumFFTSize-1)))
	}
	s.SetSampleRate(sampleRate)
	zeros := make([]float32, SpectrumBins)
	s.display.Store(&zeros)
	return s
}

// SetSampleRate resets the analyzer when the audio format changes.
func (s *Spectrum) SetSampleRate(sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if s.sampleRate != sampleRate {
		s.sampleRate = sampleRate
		s.Reset()
	}
}

// Snapshot returns the latest normalized 0..1 spectrum bars.
func (s *Spectrum) Snapshot() []float32 {
	p := s.display.Load()
	if p == nil {
		return nil
	}
	return *p
}

// Push feeds one stereo frame into the analyzer.
func (s *Spectrum) Push(l, r float64) {
	s.left[s.pos] = l
	s.right[s.pos] = r
	s.pos++
	if s.pos >= spectrumFFTSize {
		s.analyze()
		s.pos = 0
	}
}

// Reset clears the input buffers and published bars.
func (s *Spectrum) Reset() {
	clear(s.left)
	clear(s.right)
	s.pos = 0
	zeros := make([]float32, SpectrumBins)
	s.display.Store(&zeros)
}

func (s *Spectrum) analyze() {
	for i := 0; i < spectrumFFTSize; i++ {
		w := s.hann[i]
		s.re[i] = s.left[i] * w
		s.im[i] = s.right[i] * w
	}
	fft(s.re, s.im)

	old := s.Snapshot()
	out := make([]float32, SpectrumBins)
	maxBin := spectrumFFTSize/2 - 1
	for k := 0; k < SpectrumBins; k++ {
		t := float64(k) / float64(SpectrumBins-1)
		freq := 20 * math.Pow(1000, t)
		idx := int(freq * float64(spectrumFFTSize) / float64(s.sampleRate))
		if idx < 1 {
			idx = 1
		}
		if idx > maxBin {
			idx = maxBin
		}
		best := 0.0
		for j := idx - 1; j <= idx+1; j++ {
			if j < 1 || j > maxBin {
				continue
			}
			mag := math.Sqrt(s.re[j]*s.re[j]+s.im[j]*s.im[j]) * 2 / spectrumFFTSize
			if mag > best {
				best = mag
			}
		}
		db := 20 * math.Log10(best+1e-9)
		norm := Clamp((db+90)/90, 0, 1)
		if old != nil && k < len(old) {
			norm = math.Max(norm, float64(old[k])*0.82)
		}
		out[k] = float32(norm)
	}
	s.display.Store(&out)
}

// fft is an in-place iterative radix-2 Cooley-Tukey transform.
func fft(real, imag []float64) {
	n := len(real)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			real[i], real[j] = real[j], real[i]
			imag[i], imag[j] = imag[j], imag[i]
		}
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		ang := -2 * math.Pi / float64(size)
		for start := 0; start < n; start += size {
			for j := 0; j < half; j++ {
				theta := ang * float64(j)
				wr, wi := math.Cos(theta), math.Sin(theta)
				a := start + j
				b := a + half
				ur, ui := real[a], imag[a]
				vr := real[b]*wr - imag[b]*wi
				vi := real[b]*wi + imag[b]*wr
				real[a], imag[a] = ur+vr, ui+vi
				real[b], imag[b] = ur-vr, ui-vi
			}
		}
	}
}
