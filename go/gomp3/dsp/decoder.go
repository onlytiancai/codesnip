package dsp

import (
	"io"
	"sync/atomic"

	"github.com/egoist/mygo/audio"
)

// Decoder wraps an audio.Decoder and runs the whole DSP pipeline in its
// Read method. MyGo calls Read from one goroutine, so DSP state needs no locks.
type Decoder struct {
	src audio.Decoder

	config  atomic.Pointer[Pipeline]
	applied *Pipeline

	chain    *FilterChain
	surround *SurroundProcessor
	limiter  *Limiter
	spectrum *Spectrum

	left, right []float64
	mono        []float32
}

// NewDecoder wraps src with a neutral or provided pipeline.
func NewDecoder(src audio.Decoder, p Pipeline) *Decoder {
	clone := p.Clone()
	d := &Decoder{
		src:      src,
		chain:    NewFilterChain(clone, src.SampleRate()),
		surround: NewSurroundProcessor(),
		limiter:  NewLimiter(src.SampleRate(), p.FX.Limiter),
		spectrum: NewSpectrum(src.SampleRate()),
	}
	d.applied = &clone
	d.config.Store(&clone)
	d.surround.Update(clone.Surround, src.SampleRate())
	return d
}

// SetPipeline publishes a new immutable DSP snapshot to the audio goroutine.
func (d *Decoder) SetPipeline(p Pipeline) {
	clone := p.Clone()
	d.config.Store(&clone)
}

// Pipeline returns the most recently published snapshot.
func (d *Decoder) Pipeline() Pipeline {
	p := d.config.Load()
	if p == nil {
		return DefaultPipeline()
	}
	return p.Clone()
}

// Spectrum returns the analyzer fed by this decoder.
func (d *Decoder) Spectrum() *Spectrum { return d.spectrum }

// SampleRate reports the source sample rate.
func (d *Decoder) SampleRate() int { return d.src.SampleRate() }

// Channels reports two channels: mono sources are duplicated.
func (d *Decoder) Channels() int { return 2 }

// Len reports the source length in frames.
func (d *Decoder) Len() int64 { return d.src.Len() }

// SetPosition seeks the source and clears all DSP history.
func (d *Decoder) SetPosition(frame int64) error {
	if err := d.src.SetPosition(frame); err != nil {
		return err
	}
	d.resetDSP()
	return nil
}

// Read decodes and processes frames into the interleaved stereo buffer.
func (d *Decoder) Read(buf []float32) (int, error) {
	cfg := d.config.Load()
	if cfg == nil {
		defaults := DefaultPipeline()
		cfg = &defaults
	}
	sr := d.SampleRate()
	if cfg != d.applied {
		d.chain.Update(*cfg, sr)
		d.surround.Update(cfg.Surround, sr)
		d.limiter.Update(cfg.FX.Limiter, sr)
		d.spectrum.SetSampleRate(sr)
		d.applied = cfg
	}

	frames := len(buf) / 2
	if frames <= 0 {
		return 0, nil
	}
	d.ensureScratch(frames)

	var decodedFrames int
	var err error
	if d.src.Channels() == 2 {
		var n int
		n, err = d.src.Read(buf[:frames*2])
		decodedFrames = n / 2
	} else {
		var n int
		n, err = d.src.Read(d.mono[:frames])
		decodedFrames = n
		for i := 0; i < decodedFrames; i++ {
			s := d.mono[i]
			buf[i*2], buf[i*2+1] = s, s
		}
	}
	if decodedFrames <= 0 {
		return 0, err
	}

	l, r := d.left[:decodedFrames], d.right[:decodedFrames]
	for i := 0; i < decodedFrames; i++ {
		l[i] = float64(buf[i*2])
		r[i] = float64(buf[i*2+1])
	}

	preamp := DBToGain(cfg.FX.PreampDB)
	boost := Clamp(cfg.FX.Boost, 1, 2)
	for i := range l {
		l[i] *= preamp
		r[i] *= preamp
	}

	d.chain.ProcessBlock(l, r)
	d.surround.ProcessBlock(l, r)

	for i := range l {
		lv, rv := ApplyBalance(l[i], r[i], cfg.FX.Balance, cfg.FX.LeftOn, cfg.FX.RightOn)
		lv *= boost
		rv *= boost
		lv, rv = d.limiter.Process(lv, rv)
		l[i], r[i] = lv, rv
		d.spectrum.Push(lv, rv)
		buf[i*2] = float32(lv)
		buf[i*2+1] = float32(rv)
	}

	return decodedFrames * 2, err
}

// Close closes the underlying file when supported.
func (d *Decoder) Close() error {
	if c, ok := d.src.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func (d *Decoder) resetDSP() {
	d.chain.Reset()
	d.surround.Reset()
	d.limiter.Reset()
	d.spectrum.Reset()
}

func (d *Decoder) ensureScratch(frames int) {
	if cap(d.left) < frames {
		d.left = make([]float64, frames)
		d.right = make([]float64, frames)
	} else {
		d.left = d.left[:frames]
		d.right = d.right[:frames]
	}
	if cap(d.mono) < frames {
		d.mono = make([]float32, frames)
	} else {
		d.mono = d.mono[:frames]
	}
}
