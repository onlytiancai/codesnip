package dsp

const surroundMaxDelay = 8192

// SurroundProcessor widens a stereo image for headphones without changing
// the macOS system output or requiring a virtual audio device.
type SurroundProcessor struct {
	enabled    bool
	strength   float64
	lowFreq    float64
	sampleRate int
	lowL, lowR *Biquad

	delayLR []float64
	delayRL []float64
	index   int
	wet     float64
	tap1    int
	tap2    int
}

// NewSurroundProcessor creates a bypassed processor.
func NewSurroundProcessor() *SurroundProcessor {
	s := &SurroundProcessor{
		delayLR: make([]float64, surroundMaxDelay),
		delayRL: make([]float64, surroundMaxDelay),
	}
	s.Update(Surround{}, 48000)
	return s
}

// Update applies a new surround snapshot. Major changes reset the delay lines.
func (s *SurroundProcessor) Update(cfg Surround, sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	strength := Clamp(cfg.Strength, 0, 1)
	lowFreq := Clamp(cfg.LowFreq, 0, 1)
	enabled := cfg.Enabled

	if s.sampleRate != 0 && s.sampleRate != sampleRate {
		s.Reset()
	}
	if enabled != s.enabled {
		if !enabled {
			s.Reset()
		}
		s.enabled = enabled
	}
	s.strength = strength
	s.lowFreq = lowFreq
	s.sampleRate = sampleRate

	lowGain := lowFreq * 6
	lowCoeff := DesignCoefficients(LowShelf, 90, lowGain, 0.707, float64(sampleRate))
	if s.lowL == nil {
		s.lowL = NewBiquad(lowCoeff)
		s.lowR = NewBiquad(lowCoeff)
	} else {
		s.lowL.SetCoefficients(lowCoeff)
		s.lowR.SetCoefficients(lowCoeff)
	}

	s.tap1 = int((0.008 + 0.010*strength) * float64(sampleRate))
	s.tap2 = int(float64(s.tap1) * 2.4)
	s.wet = 0.22 * strength
}

// ProcessBlock applies low-frequency emphasis, M/S widening and short
// cross-channel decorrelation. The processor is bypassed when disabled.
func (s *SurroundProcessor) ProcessBlock(l, r []float64) {
	if !s.enabled || s.strength <= 0 && s.lowFreq <= 0 {
		return
	}
	width := 1 + 1.2*s.strength
	for i := range l {
		li := s.lowL.Process(l[i])
		ri := s.lowR.Process(r[i])

		delayR1 := readDelay(s.delayRL, s.index, s.tap1)
		delayL1 := readDelay(s.delayLR, s.index, s.tap1)
		delayR2 := readDelay(s.delayRL, s.index, s.tap2)
		delayL2 := readDelay(s.delayLR, s.index, s.tap2)

		s.delayLR[s.index] = li
		s.delayRL[s.index] = ri
		s.index++
		if s.index >= len(s.delayLR) {
			s.index = 0
		}

		mid := (li + ri) * 0.5
		side := (li - ri) * 0.5 * width

		l[i] = mid + side + s.wet*delayR1 + s.wet*0.35*delayR2
		r[i] = mid - side + s.wet*delayL1 + s.wet*0.35*delayL2
	}
}

// Reset clears the delay lines and shelf filters.
func (s *SurroundProcessor) Reset() {
	for i := range s.delayLR {
		s.delayLR[i] = 0
		s.delayRL[i] = 0
	}
	s.index = 0
	if s.lowL != nil {
		s.lowL.Reset()
	}
	if s.lowR != nil {
		s.lowR.Reset()
	}
}

func readDelay(buf []float64, index, delay int) float64 {
	if delay <= 0 {
		return 0
	}
	if delay >= len(buf) {
		delay = len(buf) - 1
	}
	pos := index - delay
	if pos < 0 {
		pos += len(buf)
	}
	return buf[pos]
}
