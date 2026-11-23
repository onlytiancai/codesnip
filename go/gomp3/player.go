package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/audio"
	_ "github.com/egoist/mygo/audio/mp3"

	"gomp3/dsp"
)

// PlayerInfo describes the loaded track for the UI.
type PlayerInfo struct {
	Path       string
	Name       string
	Format     string
	SampleRate int
	Channels   int
	Duration   time.Duration
	FileSize   int64
	Error      string
}

// Player owns the audio player and its DSP decoder.
type Player struct {
	state      *audio.Player
	decoder    *dsp.Decoder
	file       *os.File
	info       PlayerInfo
	volume     float64
	onChange   func()
	nowPlaying func()
}

// NewPlayer creates an empty player. onChange is called for state changes.
func NewPlayer(onChange func()) *Player {
	audio.SetLatency(audio.LatencyPlayback)
	return &Player{volume: 1, onChange: onChange}
}

// Open closes the current track and loads path.
func (p *Player) Open(path string, pipeline dsp.Pipeline) error {
	p.Close()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	raw, format, err := audio.NewDecoder(f)
	if err != nil {
		f.Close()
		return err
	}
	dec := dsp.NewDecoder(raw, pipeline)
	state := audio.NewPlayer(dec)
	state.SetVolume(p.volume)

	st, statErr := f.Stat()
	size := int64(0)
	if statErr == nil {
		size = st.Size()
	}
	duration := state.Duration()
	p.file = f
	p.decoder = dec
	p.state = state
	p.info = PlayerInfo{
		Path:       path,
		Name:       strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Format:     strings.ToUpper(format),
		SampleRate: dec.SampleRate(),
		Channels:   dec.Channels(),
		Duration:   duration,
		FileSize:   size,
	}
	if p.nowPlaying != nil {
		p.nowPlaying()
	}
	state.OnEnd(func() {
		if p.onChange != nil {
			p.onChange()
		}
	})
	p.notify()
	return nil
}

// Close releases the current audio player and file.
func (p *Player) Close() {
	if p.nowPlaying != nil {
		p.nowPlaying()
	}
	if p.state != nil {
		p.state.Close()
	}
	p.state = nil
	p.decoder = nil
	if p.file != nil {
		p.file.Close()
	}
	p.file = nil
	p.info = PlayerInfo{}
	p.notify()
}

// Loaded reports whether a track is open.
func (p *Player) Loaded() bool { return p.state != nil && p.decoder != nil }

// Info returns a copy of the current track information.
func (p *Player) Info() PlayerInfo { return p.info }

// Playing reports whether playback is active.
func (p *Player) Playing() bool { return p.state != nil && p.state.Playing() }

// Play starts playback.
func (p *Player) Play() {
	if p.state == nil {
		return
	}
	p.state.Play()
	p.syncNowPlaying()
	p.notify()
}

// Pause pauses playback.
func (p *Player) Pause() {
	if p.state == nil {
		return
	}
	p.state.Pause()
	p.syncNowPlaying()
	p.notify()
}

// TogglePlay pauses or starts playback.
func (p *Player) TogglePlay() {
	if p.Playing() {
		p.Pause()
	} else {
		p.Play()
	}
}

// Stop returns to the start and pauses.
func (p *Player) Stop() {
	if p.state == nil {
		return
	}
	p.state.Stop()
	p.syncNowPlaying()
	p.notify()
}

// Position returns the heard playback position.
func (p *Player) Position() time.Duration {
	if p.state == nil {
		return 0
	}
	return p.state.Position()
}

// Duration returns the loaded track duration.
func (p *Player) Duration() time.Duration {
	if p.state == nil {
		return 0
	}
	return p.state.Duration()
}

// Seek moves playback to a duration.
func (p *Player) Seek(t time.Duration) {
	if p.state == nil {
		return
	}
	if t < 0 {
		t = 0
	}
	if d := p.Duration(); d > 0 && t > d {
		t = d
	}
	p.state.Seek(t)
	p.syncNowPlaying()
	p.notify()
}

// SetVolume sets the player's normal 0..1 gain. Boost is handled in DSP.
func (p *Player) SetVolume(v float64) {
	p.volume = clamp01(v)
	if p.state != nil {
		p.state.SetVolume(p.volume)
	}
}

// Volume returns the normal 0..1 gain.
func (p *Player) Volume() float64 { return p.volume }

// SetPipeline publishes new DSP settings. The decoder ramps coefficients and
// gains over a few milliseconds, so we deliberately do not seek here: seeking
// on every slider change would interrupt playback and click. New settings are
// heard as soon as the already-buffered frames drain (~tens of ms).
func (p *Player) SetPipeline(pipeline dsp.Pipeline) {
	if p.decoder == nil {
		return
	}
	p.decoder.SetPipeline(pipeline)
	p.notify()
}

// Spectrum returns the analyzer fed by the active decoder.
func (p *Player) Spectrum() *dsp.Spectrum {
	if p.decoder == nil {
		return nil
	}
	return p.decoder.Spectrum()
}

// EnableSystemMediaKeys connects the player to macOS media keys.
func (p *Player) EnableSystemMediaKeys() {
	if p.nowPlaying != nil || p.state == nil {
		return
	}
	p.nowPlaying = mygo.NowPlaying.Control(p.state)
	p.syncNowPlaying()
}

func (p *Player) syncNowPlaying() {
	if p.nowPlaying == nil || p.state == nil {
		return
	}
	state := mygo.PlaybackPaused
	if p.Playing() {
		state = mygo.PlaybackPlaying
	}
	mygo.NowPlaying.Set(mygo.NowPlayingInfo{
		Title:    p.info.Name,
		Duration: p.info.Duration,
		Position: p.Position(),
		State:    state,
	})
}

func (p *Player) notify() {
	if p.onChange != nil {
		p.onChange()
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// FormatTime renders a duration as m:ss or h:m:ss.
func FormatTime(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d.Seconds())
	hours := total / 3600
	minutes := (total % 3600) / 60
	seconds := total % 60
	if hours > 0 {
		return fmt.Sprintf("%d:%02d:%02d", hours, minutes, seconds)
	}
	return fmt.Sprintf("%d:%02d", minutes, seconds)
}
