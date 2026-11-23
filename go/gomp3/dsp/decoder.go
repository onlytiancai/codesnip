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

	// Gains are smoothed per sample so balance / channel / preamp / boost
	// edits never produce a discontinuity (the "click" noise).
	smPreamp  smoother
	smBoost   smoother
	smBalance smoother
	smLeft    smoother
	smRight   smoother

	left, right []float64
	mono        []float32
}

// NewDecoder wraps src with a neutral or provided pipeline.
func NewDecoder(src audio.Decoder, p Pipeline) *Decoder {
	clone := p.Clone()
	sr := float64(src.SampleRate())
	lg, rg := channelGains(clone.FX)
	d := &Decoder{
		src:       src,
		chain:     NewFilterChain(clone, src.SampleRate()),
		surround:  NewSurroundProcessor(),
		limiter:   NewLimiter(src.SampleRate(), p.FX.Limiter),
		spectrum:  NewSpectrum(src.SampleRate()),
		smPreamp:  newSmoother(DBToGain(clone.FX.PreampDB), sr),
		smBoost:   newSmoother(Clamp(clone.FX.Boost, 1, 2), sr),
		smBalance: newSmoother(Clamp(clone.FX.Balance, -1, 1), sr),
		smLeft:    newSmoother(lg, sr),
		smRight:   newSmoother(rg, sr),
	}
	d.applied = &clone
	d.config.Store(&clone)
	d.surround.Update(clone.Surround, src.SampleRate())
	return d
}

// channelGains maps the channel switches to 0/1 gain targets.
func channelGains(fx Effects) (float64, float64) {
	lg, rg := 1.0, 1.0
	if !fx.LeftOn {
		lg = 0
	}
	if !fx.RightOn {
		rg = 0
	}
	return lg, rg
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
		// Retarget the smoothed gains; the audio thread ramps to them below.
		d.smPreamp.set(DBToGain(cfg.FX.PreampDB))
		d.smBoost.set(Clamp(cfg.FX.Boost, 1, 2))
		d.smBalance.set(Clamp(cfg.FX.Balance, -1, 1))
		lg, rg := channelGains(cfg.FX)
		d.smLeft.set(lg)
		d.smRight.set(rg)
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

	// Preamp is applied per sample with smoothing so it cannot click.
	for i := range l {
		g := d.smPreamp.next()
		l[i] *= g
		r[i] *= g
	}

	d.chain.ProcessBlock(l, r)
	d.surround.ProcessBlock(l, r)

	for i := range l {
		// Balance, boost and the channel switches all ramp smoothly. The
		// channel switches are folded into the per-channel gain here instead
		// of being hard 0/1 multipliers inside ApplyBalance.
		balance := d.smBalance.next()
		boost := d.smBoost.next()
		lv, rv := ApplyBalance(l[i], r[i], balance, true, true)
		lv *= boost * d.smLeft.next()
		rv *= boost * d.smRight.next()
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
	// Snap the smoothed gains to their targets: after a real seek we want the
	// new position to play at the current settings immediately.
	d.smPreamp.current = d.smPreamp.target
	d.smBoost.current = d.smBoost.target
	d.smBalance.current = d.smBalance.target
	d.smLeft.current = d.smLeft.target
	d.smRight.current = d.smRight.target
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
