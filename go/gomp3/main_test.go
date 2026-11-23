package main

import (
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

func testApp(t *testing.T) *app {
	t.Helper()
	t.Setenv("GOMP3_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	return newApp()
}

func TestViewSwitchesToExpertAndAddsBand(t *testing.T) {
	a := testApp(t)
	tt := ui.NewTester(a.view, 1080, 760)

	if err := tt.Click("专家"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()
	if a.model.EQTab != 2 {
		t.Fatalf("EQ tab = %d, want 2", a.model.EQTab)
	}

	if err := tt.Click("+ 添加波段"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()
	if len(a.model.Pipeline.Params) != 1 {
		t.Fatalf("parametric bands = %d, want 1", len(a.model.Pipeline.Params))
	}
	if !a.model.PresetModified {
		t.Fatal("adding a band should mark the preset modified")
	}
}

func TestApplyNoneThenSwitchModeKeepsEffectsOff(t *testing.T) {
	a := testApp(t)
	a.applyPreset(PresetNone)
	if a.model.Pipeline.Basic.Enabled || a.model.Pipeline.Fixed10.Enabled || a.model.Pipeline.Fixed31.Enabled || a.model.Pipeline.Surround.Enabled {
		t.Fatalf("none preset did not disable effects: %+v", a.model.Pipeline)
	}
	a.setAdvancedMode(1)
	if a.model.Pipeline.Fixed10.Enabled || a.model.Pipeline.Fixed31.Enabled {
		t.Fatal("switching EQ mode should not enable graphic EQ under the 无 preset")
	}
}

func TestResetPanels(t *testing.T) {
	a := testApp(t)
	// Dirty every panel.
	a.model.Pipeline.Basic.Bass = 9
	a.model.Pipeline.FX.PreampDB = 6
	a.model.Pipeline.Fixed10.Gains[0] = 7
	a.model.Pipeline.Params = append(a.model.Pipeline.Params, dsp.NewParametricBand())
	a.model.Pipeline.Surround.Enabled = true
	a.model.Pipeline.Surround.Strength = 0.8

	a.resetPanel(0)
	if a.model.Pipeline.Basic.Bass != 0 || a.model.Pipeline.FX.PreampDB != 0 {
		t.Fatal("basic reset did not restore defaults")
	}
	a.resetPanel(1)
	if a.model.Pipeline.Fixed10.Gains[0] != 0 {
		t.Fatal("advanced reset did not clear graphic EQ gains")
	}
	a.resetPanel(2)
	if len(a.model.Pipeline.Params) != 0 {
		t.Fatal("expert reset did not clear parametric bands")
	}
	a.resetPanel(3)
	def := dsp.DefaultPipeline()
	if a.model.Pipeline.Surround.Enabled || a.model.Pipeline.Surround.Strength != def.Surround.Strength {
		t.Fatal("surround reset did not restore defaults")
	}
}

func TestViewSavesCustomPreset(t *testing.T) {
	a := testApp(t)
	tt := ui.NewTester(a.view, 1080, 760)
	if err := tt.Click("保存"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()
	a.model.PresetName = "My Test EQ"
	if err := tt.Click("确定保存"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()

	if len(a.cfg.CustomPresets) != 1 || a.cfg.CustomPresets[0].Name != "My Test EQ" {
		t.Fatalf("custom presets = %+v", a.cfg.CustomPresets)
	}
	if a.model.ActivePreset != "My Test EQ" {
		t.Fatalf("active preset = %q", a.model.ActivePreset)
	}
}
