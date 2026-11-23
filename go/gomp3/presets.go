package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gomp3/dsp"
)

const configVersion = 1

// PresetNone disables every EQ and effect, restoring a flat, unprocessed sound.
const PresetNone = "无"

// Preset stores a complete snapshot of the DSP settings.
type Preset struct {
	Name     string       `json:"name"`
	Builtin  bool         `json:"builtin"`
	Pipeline dsp.Pipeline `json:"pipeline"`
}

// Config is the versioned on-disk user configuration.
type Config struct {
	Version       int      `json:"version"`
	ActivePreset  string   `json:"active_preset"`
	CustomPresets []Preset `json:"custom_presets"`
	LastFile      string   `json:"last_file"`
	EQTab         int      `json:"eq_tab"`
	AdvancedMode  int      `json:"advanced_mode"`
}

func defaultConfig() Config {
	return Config{
		Version:       configVersion,
		ActivePreset:  "Flat",
		CustomPresets: []Preset{},
		EQTab:         0,
		AdvancedMode:  0,
	}
}

func configPath() (string, error) {
	if path := os.Getenv("GOMP3_CONFIG"); path != "" {
		return path, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "gomp3", "config.json"), nil
}

// LoadConfig reads the user config, falling back to defaults if missing or bad.
func LoadConfig() Config {
	cfg := defaultConfig()
	path, err := configPath()
	if err != nil {
		return cfg
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg
	}
	var loaded Config
	if err := json.Unmarshal(data, &loaded); err != nil {
		return cfg
	}
	if loaded.Version != configVersion {
		return cfg
	}
	if loaded.CustomPresets == nil {
		loaded.CustomPresets = []Preset{}
	}
	for i := range loaded.CustomPresets {
		loaded.CustomPresets[i].Builtin = false
	}
	cfg = loaded
	return cfg
}

// SaveConfig writes the user config atomically.
func SaveConfig(cfg Config) error {
	path, err := configPath()
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "config-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// NonePipeline returns a completely flat pipeline with every EQ and effect
// switched off. The limiter stays on only as a safety net against clipping.
func NonePipeline() dsp.Pipeline {
	p := dsp.DefaultPipeline()
	p.Basic.Enabled = false
	p.Basic.Bass, p.Basic.Mid, p.Basic.Treble = 0, 0, 0
	p.Fixed10.Enabled = false
	p.Fixed31.Enabled = false
	p.Fixed10.Gains = [10]float64{}
	p.Fixed31.Gains = [31]float64{}
	p.Params = []dsp.ParametricBand{}
	p.Surround = dsp.Surround{Enabled: false, LowFreq: 0, Strength: 0}
	p.FX.PreampDB = 0
	p.FX.Boost = 1
	p.FX.Balance = 0
	p.FX.LeftOn, p.FX.RightOn = true, true
	p.FX.Limiter = true
	return p
}

// BuiltinPresets returns the factory presets.
func BuiltinPresets() []Preset {
	type spec struct {
		name  string
		gains [10]float64
		bass  float64
		mid   float64
		treb  float64
		boost float64
		lower float64
	}
	specs := []spec{
		{"Flat", [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0, 0, 0, 1, 0},
		{"Pop", [10]float64{-1, -0.5, 0, 1.5, 3, 2, 0.5, -0.5, -1.5, -2}, 2, 1, 1.5, 1, 0},
		{"Classical", [10]float64{2, 1.5, 1, 0.5, 0, 0, 0.5, 1.5, 2, 2.5}, 1.5, 0, 1, 1, 0},
		{"Movie", [10]float64{4, 3, 1.5, 0.5, -0.5, 0, 1, 2.5, 4, 4.5}, 4, -1, 2, 1.05, 0.3},
		{"Hip Hop", [10]float64{5, 4, 2.5, 0.5, -1, -0.5, 0.5, 1.5, 2.5, 3}, 5, -1, 1, 1.05, 0.4},
		{"Jazz", [10]float64{2, 1, 0.5, 1, 1.5, 1, 0.5, 1, 1.5, 2}, 2, 0.5, 1.5, 1, 0},
		{"Electronic", [10]float64{3.5, 3, 1.5, 0, -1, 0.5, 2, 3, 3.5, 3}, 4, -0.5, 2, 1.05, 0.2},
		{"Rock", [10]float64{3, 2, 0.5, -1, -1.5, 0, 1.5, 2.5, 3, 3}, 3, -1, 2, 1.05, 0.15},
		{"Bass Boost", [10]float64{6, 5, 3.5, 1.5, 0, 0, 0, 0, 0, 0}, 6, 0, 0, 1.05, 0.5},
		{"Vocal", [10]float64{-2, -1.5, -1, 0, 2, 3.5, 3, 1.5, -1, -2}, -1, 2, 0.5, 1, 0},
	}

	out := make([]Preset, 0, len(specs)+1)
	out = append(out, Preset{Name: PresetNone, Builtin: true, Pipeline: NonePipeline()})
	for _, s := range specs {
		p := dsp.DefaultPipeline()
		p.Basic.Enabled = true
		p.Basic.Bass = s.bass
		p.Basic.Mid = s.mid
		p.Basic.Treble = s.treb
		p.Fixed10.Gains = s.gains
		p.Fixed31.Gains = expand10to31(s.gains)
		p.FX.Boost = s.boost
		p.Surround.LowFreq = s.lower
		out = append(out, Preset{Name: s.name, Builtin: true, Pipeline: p})
	}
	return out
}

// PresetNames lists built-in and custom names for the preset selector.
func (c *Config) PresetNames() []string {
	names := make([]string, 0, len(BuiltinPresets())+len(c.CustomPresets))
	for _, p := range BuiltinPresets() {
		names = append(names, p.Name)
	}
	for _, p := range c.CustomPresets {
		names = append(names, p.Name)
	}
	sort.Strings(names)
	return names
}

// FindPreset looks up a preset by exact name.
func (c *Config) FindPreset(name string) (Preset, bool) {
	for _, p := range BuiltinPresets() {
		if p.Name == name {
			return p, true
		}
	}
	for _, p := range c.CustomPresets {
		if p.Name == name {
			return p, true
		}
	}
	return Preset{}, false
}

// SavePreset adds or replaces a custom preset and persists the config.
func (c *Config) SavePreset(name string, pipeline dsp.Pipeline, replace bool) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("预设名称不能为空")
	}
	for _, p := range BuiltinPresets() {
		if p.Name == name {
			return errors.New("内置预设名称不能重复使用")
		}
	}
	for i := range c.CustomPresets {
		if c.CustomPresets[i].Name == name {
			if !replace {
				return errors.New("同名预设已存在")
			}
			c.CustomPresets[i].Pipeline = pipeline.Clone()
			c.ActivePreset = name
			return SaveConfig(*c)
		}
	}
	c.CustomPresets = append(c.CustomPresets, Preset{
		Name:     name,
		Builtin:  false,
		Pipeline: pipeline.Clone(),
	})
	c.ActivePreset = name
	return SaveConfig(*c)
}

// DeletePreset removes a custom preset.
func (c *Config) DeletePreset(name string) error {
	for i, p := range c.CustomPresets {
		if p.Name == name {
			c.CustomPresets = append(c.CustomPresets[:i], c.CustomPresets[i+1:]...)
			if c.ActivePreset == name {
				c.ActivePreset = "Flat"
			}
			return SaveConfig(*c)
		}
	}
	return errors.New("只能删除自定义预设")
}

// ApplyPresetMode enables the graphic EQ matching the current advanced view.
func ApplyPresetMode(p dsp.Pipeline, advancedMode int) dsp.Pipeline {
	p.Fixed10.Enabled = advancedMode == 0
	p.Fixed31.Enabled = advancedMode == 1
	return p
}

// expand10to31 interpolates ten graphic-EQ gains onto third-octave centers.
func expand10to31(g [10]float64) [31]float64 {
	var out [31]float64
	for i, f := range dsp.Fixed31Freqs {
		if f <= dsp.Fixed10Freqs[0] {
			out[i] = g[0]
			continue
		}
		if f >= dsp.Fixed10Freqs[9] {
			out[i] = g[9]
			continue
		}
		for j := 0; j < 9; j++ {
			lo, hi := dsp.Fixed10Freqs[j], dsp.Fixed10Freqs[j+1]
			if f >= lo && f <= hi {
				t := (math.Log(f) - math.Log(lo)) / (math.Log(hi) - math.Log(lo))
				out[i] = g[j]*(1-t) + g[j+1]*t
				break
			}
		}
	}
	return out
}
