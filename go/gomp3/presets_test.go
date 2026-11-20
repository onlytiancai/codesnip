package main

import (
	"path/filepath"
	"testing"

	"gomp3/dsp"
)

func TestBuiltinPresetsIncludeRequiredNames(t *testing.T) {
	want := []string{"Pop", "Classical", "Movie", "Hip Hop", "Jazz", "Electronic"}
	presets := BuiltinPresets()
	for _, name := range want {
		found := false
		for _, p := range presets {
			if p.Name == name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing built-in preset %q", name)
		}
	}
}

func TestExpand10to31(t *testing.T) {
	var g [10]float64
	for i := range g {
		g[i] = float64(i)
	}
	out := expand10to31(g)
	if len(out) != 31 {
		t.Fatalf("31-band length = %d", len(out))
	}
	if out[0] != 0 || out[30] != 9 {
		t.Fatalf("endpoints = %.1f, %.1f", out[0], out[30])
	}
}

func TestCustomPresetCRUD(t *testing.T) {
	t.Setenv("GOMP3_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	cfg := defaultConfig()
	p := dsp.DefaultPipeline()
	p.Fixed10.Gains[0] = 3.5

	if err := cfg.SavePreset("My EQ", p, false); err != nil {
		t.Fatal(err)
	}
	loaded := LoadConfig()
	if len(loaded.CustomPresets) != 1 || loaded.CustomPresets[0].Name != "My EQ" {
		t.Fatalf("loaded presets = %+v", loaded.CustomPresets)
	}
	if loaded.CustomPresets[0].Pipeline.Fixed10.Gains[0] != 3.5 {
		t.Fatal("preset pipeline was not persisted")
	}

	if err := cfg.SavePreset("My EQ", p, false); err == nil {
		t.Fatal("duplicate custom preset should fail")
	}
	if err := cfg.SavePreset("My EQ", p, true); err != nil {
		t.Fatal(err)
	}
	if err := cfg.DeletePreset("My EQ"); err != nil {
		t.Fatal(err)
	}
	if len(cfg.CustomPresets) != 0 {
		t.Fatalf("preset was not deleted: %+v", cfg.CustomPresets)
	}
}

func TestApplyPresetMode(t *testing.T) {
	p := dsp.DefaultPipeline()
	p = ApplyPresetMode(p, 1)
	if p.Fixed10.Enabled || !p.Fixed31.Enabled {
		t.Fatal("mode 1 should enable 31-band EQ only")
	}
	p = ApplyPresetMode(p, 0)
	if !p.Fixed10.Enabled || p.Fixed31.Enabled {
		t.Fatal("mode 0 should enable 10-band EQ only")
	}
}
