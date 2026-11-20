package ui

import (
	"time"

	"gomp3/dsp"
)

// Model is the UI-facing state. Main owns it and updates it on the main
// thread; callbacks forward user actions to the player and config store.
type Model struct {
	Pipeline dsp.Pipeline

	PresetNames     []string
	ActivePreset    string
	PresetModified  bool
	PresetModalOpen bool
	PresetName      string

	EQTab        int // 0 basic, 1 advanced, 2 expert, 3 surround
	AdvancedMode int // 0 = 10 band, 1 = 31 band
	SampleRate   int

	FileName string
	FileMeta string
	Position time.Duration
	Duration time.Duration
	Volume   float64
	Playing  bool
	Status   string
	Spectrum *dsp.Spectrum

	OnOpen         func()
	OnPlayToggle   func()
	OnStop         func()
	OnSeek         func(time.Duration)
	OnVolume       func(float64)
	OnPreset       func(string)
	OnSavePreset   func(string)
	OnDeletePreset func()
	OnPipeline     func()
	OnAdvancedMode func(int)
}

// MarkDirty flags the active preset as edited and publishes the DSP snapshot.
func (m *Model) MarkDirty() {
	m.PresetModified = true
	if m.OnPipeline != nil {
		m.OnPipeline()
	}
}
