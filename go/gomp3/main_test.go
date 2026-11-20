package main

import (
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"
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
