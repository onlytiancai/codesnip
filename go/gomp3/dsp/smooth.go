package dsp

// smoothSeconds is the ramp time used to avoid clicks when a gain or filter
// coefficient changes while audio is playing.
const smoothSeconds = 0.015

// smoother is a per-sample de-zipper. It walks current towards target by at
// most step each call, turning an abrupt parameter jump into a short ramp so
// the signal stays continuous and no click is heard.
type smoother struct {
	current float64
	target  float64
	step    float64
}

func newSmoother(value, sampleRate float64) smoother {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	return smoother{
		current: value,
		target:  value,
		step:    1 / (smoothSeconds * sampleRate),
	}
}

func (s *smoother) set(target float64) { s.target = target }

func (s *smoother) next() float64 {
	d := s.target - s.current
	switch {
	case d > s.step:
		s.current += s.step
	case d < -s.step:
		s.current -= s.step
	default:
		s.current = s.target
	}
	return s.current
}
