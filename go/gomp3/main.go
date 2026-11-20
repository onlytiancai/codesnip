package main

import (
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
	gomp3ui "gomp3/ui"
)

type app struct {
	win     *mygo.Window
	cfg     Config
	player  *Player
	model   gomp3ui.Model
	publish func(func())
}

func newApp() *app {
	cfg := LoadConfig()
	a := &app{cfg: cfg}
	a.publish = func(fn func()) {
		if a.win != nil {
			a.win.Update(fn)
			return
		}
		fn()
	}

	pipeline := dsp.DefaultPipeline()
	if preset, ok := cfg.FindPreset(cfg.ActivePreset); ok {
		pipeline = ApplyPresetMode(preset.Pipeline, cfg.AdvancedMode)
	}

	a.model = gomp3ui.Model{
		Pipeline:       pipeline,
		PresetNames:    cfg.PresetNames(),
		ActivePreset:   cfg.ActivePreset,
		EQTab:          cfg.EQTab,
		AdvancedMode:   cfg.AdvancedMode,
		SampleRate:     44100,
		Volume:         1,
		Status:         "就绪 · 打开 MP3 开始",
		OnOpen:         a.openFile,
		OnPlayToggle:   a.togglePlay,
		OnStop:         a.stop,
		OnSeek:         a.seek,
		OnVolume:       a.setVolume,
		OnPreset:       a.applyPreset,
		OnSavePreset:   a.savePreset,
		OnDeletePreset: a.deletePreset,
		OnPipeline:     a.applyPipeline,
		OnAdvancedMode: a.setAdvancedMode,
	}

	a.player = NewPlayer(func() {
		a.publish(func() { a.syncModel() })
	})
	a.syncModel()
	return a
}

func (a *app) view(c *ui.Context) {
	gomp3ui.View(&a.model)(c)
}

func (a *app) syncModel() {
	if a.player == nil {
		return
	}
	info := a.player.Info()
	a.model.FileName = info.Name
	a.model.FileMeta = fileMeta(info)
	a.model.Position = a.player.Position()
	a.model.Duration = a.player.Duration()
	a.model.Playing = a.player.Playing()
	a.model.Volume = a.player.Volume()
	a.model.Spectrum = a.player.Spectrum()
}

func (a *app) openFile() {
	go func() {
		paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
			Parent:      a.win,
			Title:       "打开 MP3",
			Multiple:    false,
			Filters:     []mygo.FileFilter{{Name: "MP3 音频", Extensions: []string{"mp3"}}},
			ButtonLabel: "打开",
		})
		if err != nil || len(paths) == 0 {
			return
		}
		a.publish(func() { a.openPath(paths[0]) })
	}()
}

func (a *app) openPath(path string) {
	if err := a.player.Open(path, a.model.Pipeline); err != nil {
		a.model.Status = "打开失败：" + err.Error()
		return
	}
	a.player.EnableSystemMediaKeys()
	a.cfg.LastFile = path
	_ = SaveConfig(a.cfg)
	a.model.Status = fmt.Sprintf("已打开 %s", filepath.Base(path))
	a.syncModel()
}

func (a *app) togglePlay() {
	if !a.player.Loaded() {
		a.model.Status = "请先打开 MP3 文件"
		return
	}
	a.player.TogglePlay()
	a.syncModel()
}

func (a *app) stop() {
	a.player.Stop()
	a.syncModel()
}

func (a *app) seek(t time.Duration) {
	a.player.Seek(t)
	a.syncModel()
}

func (a *app) setVolume(v float64) {
	a.player.SetVolume(v)
	a.syncModel()
}

func (a *app) applyPipeline() {
	a.player.SetPipeline(a.model.Pipeline)
}

func (a *app) applyPreset(name string) {
	preset, ok := a.cfg.FindPreset(name)
	if !ok {
		return
	}
	a.cfg.ActivePreset = name
	a.model.ActivePreset = name
	a.model.Pipeline = ApplyPresetMode(preset.Pipeline, a.cfg.AdvancedMode)
	a.model.PresetModified = false
	_ = SaveConfig(a.cfg)
	a.player.SetPipeline(a.model.Pipeline)
	a.model.Status = "已应用预设：" + name
}

func (a *app) savePreset(name string) {
	pipeline := a.model.Pipeline.Clone()
	if isBuiltinPreset(name) {
		name += " 自定义"
	}
	if err := a.cfg.SavePreset(name, pipeline, true); err != nil {
		a.model.Status = "保存预设失败：" + err.Error()
		return
	}
	a.model.ActivePreset = a.cfg.ActivePreset
	a.model.PresetNames = a.cfg.PresetNames()
	a.model.PresetName = name
	a.model.PresetModified = false
	a.model.Status = "已保存预设：" + name
}

func (a *app) deletePreset() {
	name := a.model.ActivePreset
	if !a.isCustomPreset(name) {
		a.model.Status = "内置预设不能删除"
		return
	}
	if err := a.cfg.DeletePreset(name); err != nil {
		a.model.Status = "删除预设失败：" + err.Error()
		return
	}
	a.model.PresetNames = a.cfg.PresetNames()
	a.applyPreset(a.cfg.ActivePreset)
	a.model.Status = "已删除预设：" + name
}

func (a *app) setAdvancedMode(mode int) {
	if mode < 0 || mode > 1 {
		return
	}
	a.cfg.AdvancedMode = mode
	a.model.Pipeline.Fixed10.Enabled = mode == 0
	a.model.Pipeline.Fixed31.Enabled = mode == 1
	_ = SaveConfig(a.cfg)
	a.player.SetPipeline(a.model.Pipeline)
}

func isBuiltinPreset(name string) bool {
	for _, p := range BuiltinPresets() {
		if p.Name == name {
			return true
		}
	}
	return false
}

func (a *app) isCustomPreset(name string) bool {
	for _, p := range a.cfg.CustomPresets {
		if p.Name == name {
			return true
		}
	}
	return false
}

func fileMeta(info PlayerInfo) string {
	if info.Path == "" {
		return ""
	}
	size := float64(info.FileSize) / 1024 / 1024
	return fmt.Sprintf("%s · %d Hz · %d 声道 · %.1f MB · %s",
		info.Format, info.SampleRate, info.Channels, size, FormatTime(info.Duration))
}

func main() {
	a := newApp()
	mygo.App.WhenReady(func() {
		win := mygo.NewWindow(mygo.WindowOptions{
			Title:     "gomp3 · MP3 音效工作台",
			Width:     1080,
			Height:    760,
			MinWidth:  900,
			MinHeight: 620,
			StateKey:  "main",
			Content:   ui.View(a.view),
		})
		a.win = win
		go func() {
			ticker := time.NewTicker(33 * time.Millisecond)
			defer ticker.Stop()
			for range ticker.C {
				a.publish(func() {
					a.syncModel()
				})
			}
		}()
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
