package ui

import (
	"fmt"
	"time"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

// View returns a MyGo view function bound to a model.
func View(m *Model) func(c *ui.Context) {
	return func(c *ui.Context) {
		t := *c.Theme()
		t.Accent = colorGreen
		t.AccentHover = colorGreenDark
		t.AccentPressed = colorGreenDark
		t.Selection = colorGreenSoft
		c.SetTheme(&t)

		ui.Column(c).Fill().Padding(14).Gap(10).Background(colorPanel).Children(func() {
			toolbar(c, m)
			trackInfo(c, m)
			analyzer(c, m)

			ui.Segmented(c.Key("eq-tab"), &m.EQTab, "基础", "高级", "专家", "3D").Label("EQ 页面")

			ui.Box(c).Grow(1).Background(colorSurface).Border(1, colorBorder).Radius(8).Padding(12).Children(func() {
				switch m.EQTab {
				case 0:
					basicPage(c, m)
				case 1:
					advancedPage(c, m)
				case 2:
					ExpertEQ(c, m)
				case 3:
					surroundPage(c, m)
				}
			})

			ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
				status := m.Status
				if status == "" {
					status = "就绪 · 音量增强仅作用于本应用"
				}
				ui.Text(c, status).FontSize(12).TextColor(colorMuted).Grow(1)
				ui.Text(c, fmt.Sprintf("普通音量 %d%%", int(m.Volume*100))).FontSize(12).TextColor(colorMuted)
			})
			presetModal(c, m)
		})
	}
}

// analyzer guards the analyzer view when no file is open yet.
func analyzer(c *ui.Context, m *Model) {
	if m.Spectrum == nil {
		m.Spectrum = dsp.NewSpectrum(m.SampleRate)
	}
	AnalyzerView(c, m)
}

func toolbar(c *ui.Context, m *Model) {
	ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
		if ui.Button(c.Key("open"), "打开 MP3").Clicked() && m.OnOpen != nil {
			m.OnOpen()
		}
		playLabel := "播放"
		if m.Playing {
			playLabel = "暂停"
		}
		if ui.PrimaryButton(c.Key("play"), playLabel).Clicked() && m.OnPlayToggle != nil {
			m.OnPlayToggle()
		}
		if ui.Button(c.Key("stop"), "停止").Clicked() && m.OnStop != nil {
			m.OnStop()
		}
		ui.Spacer(c)
		label := m.ActivePreset
		if label == "" {
			label = "Flat"
		}
		sel := ui.Select(c.Key("preset"), &label, m.PresetNames).Width(170).Label("音效预设")
		if sel.Changed() && m.OnPreset != nil {
			m.OnPreset(label)
		}
		if ui.Button(c.Key("save-preset"), "保存").Clicked() {
			m.PresetName = label
			m.PresetModalOpen = true
		}
		if ui.Button(c.Key("delete-preset"), "删除").Clicked() && m.OnDeletePreset != nil {
			m.OnDeletePreset()
		}
		if m.PresetModified {
			ui.Text(c, "已修改").FontSize(11).TextColor(colorRed)
		}
	})
}

func trackInfo(c *ui.Context, m *Model) {
	ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
		name := m.FileName
		if name == "" {
			name = "未打开文件"
		}
		ui.Text(c, name).Bold().FontSize(15).Grow(1)
		ui.Text(c, m.FileMeta).FontSize(11).TextColor(colorMuted)
	})

	ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
		ui.Text(c, FormatDuration(m.Position)).FontSize(11).TextColor(colorMuted)
		dur := m.Duration.Seconds()
		if dur <= 0 {
			dur = 1
		}
		pos := m.Position.Seconds()
		s := ui.Slider(c.Key("progress"), &pos, 0, dur).
			Grow(1).
			Disabled(m.Duration <= 0).
			Label("播放进度")
		if s.Changed() && m.OnSeek != nil {
			m.OnSeek(time.Duration(pos * float64(time.Second)))
		}
		ui.Text(c, FormatDuration(m.Duration)).FontSize(11).TextColor(colorMuted)
	})
}

// panelHeader renders a title with a reset button that restores the panel's
// defaults through the model callback.
func panelHeader(c *ui.Context, m *Model, title string, panel int) {
	ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
		ui.Text(c, title).Bold().FontSize(14).Grow(1)
		if ui.Button(c.Key(fmt.Sprintf("reset-%d", panel)), "重置").Clicked() && m.OnReset != nil {
			m.OnReset(panel)
		}
	})
}

func basicPage(c *ui.Context, m *Model) {
	panelHeader(c, m, "基础音效", 0)
	ui.Column(c).Fill().Gap(8).Children(func() {
		ui.Row(c).Gap(12).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "前置放大器").Width(80).FontSize(13).TextColor(colorText)
			preamp := m.Pipeline.FX.PreampDB
			s := ui.Slider(c.Key("preamp"), &preamp, -12, 12).Step(0.5).Grow(1).Label("前置增益")
			if s.Changed() {
				m.Pipeline.FX.PreampDB = preamp
				m.MarkDirty()
			}
			ui.Text(c, fmt.Sprintf("%+.1f dB", preamp)).Width(74).TextAlign(ui.Center).FontSize(12)
		})

		ui.Column(c).Gap(4).Children(func() {
			Knob(c, m, "bass", "低音", &m.Pipeline.Basic.Bass, -12, 12, 0.5, "dB", nil)
			Knob(c, m, "mid", "中音", &m.Pipeline.Basic.Mid, -12, 12, 0.5, "dB", nil)
			Knob(c, m, "treble", "高音", &m.Pipeline.Basic.Treble, -12, 12, 0.5, "dB", nil)
		})

		ui.Row(c).Gap(24).AlignItems(ui.Center).Padding(4, 0).Children(func() {
			ui.Column(c).Gap(8).Grow(1).Children(func() {
				boost := m.Pipeline.FX.Boost * 100
				s := ui.Slider(c.Key("boost"), &boost, 100, 200).Step(5).Label("音量增强")
				if s.Changed() {
					m.Pipeline.FX.Boost = boost / 100
					m.MarkDirty()
				}
				ui.Text(c, fmt.Sprintf("音量增强（仅本应用） %.0f%%", boost)).FontSize(11).TextColor(colorMuted)
			})
			ui.Column(c).Gap(8).Grow(1).Children(func() {
				vol := m.Volume * 100
				sv := ui.Slider(c.Key("volume"), &vol, 0, 100).Step(1).Label("普通音量")
				if sv.Changed() && m.OnVolume != nil {
					m.OnVolume(vol / 100)
				}
				ui.Text(c, "超过 100% 会启用软限幅，请注意听力安全").FontSize(11).TextColor(colorMuted)
			})
		})

		ui.Row(c).Gap(14).AlignItems(ui.Center).Children(func() {
			if ui.Switch(c.Key("eq-on"), &m.Pipeline.Basic.Enabled).Label("三段音效").Changed() {
				m.MarkDirty()
			}
			if ui.Switch(c.Key("limiter"), &m.Pipeline.FX.Limiter).Label("峰值限制器").Changed() {
				m.MarkDirty()
			}
			if ui.Switch(c.Key("left"), &m.Pipeline.FX.LeftOn).Label("左声道").Changed() {
				m.MarkDirty()
			}
			if ui.Switch(c.Key("right"), &m.Pipeline.FX.RightOn).Label("右声道").Changed() {
				m.MarkDirty()
			}
		})
		ui.Row(c).Gap(12).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "左右平衡").Width(64).FontSize(12).TextColor(colorText)
			balance := m.Pipeline.FX.Balance
			s := ui.Slider(c.Key("balance"), &balance, -1, 1).Step(0.05).Grow(1).Label("左右平衡")
			if s.Changed() {
				m.Pipeline.FX.Balance = balance
				m.MarkDirty()
			}
			ui.Text(c, balanceLabel(balance)).Width(60).FontSize(12).TextAlign(ui.Center).TextColor(colorMuted)
		})
	})
}

func advancedPage(c *ui.Context, m *Model) {
	ui.Column(c).Fill().Gap(8).Children(func() {
		ui.Row(c).Gap(12).AlignItems(ui.Center).Children(func() {
			tabs := ui.Segmented(c.Key("eq-mode"), &m.AdvancedMode, "10 段", "31 段").Label("图形 EQ 模式")
			if tabs.Changed() && m.OnAdvancedMode != nil {
				m.OnAdvancedMode(m.AdvancedMode)
			}
			if m.AdvancedMode == 0 {
				if ui.Switch(c.Key("fixed10-on"), &m.Pipeline.Fixed10.Enabled).Label("启用 10 段 EQ").Changed() {
					m.MarkDirty()
				}
			} else {
				if ui.Switch(c.Key("fixed31-on"), &m.Pipeline.Fixed31.Enabled).Label("启用 31 段 EQ").Changed() {
					m.MarkDirty()
				}
			}
			ui.Spacer(c)
			if ui.Button(c.Key("reset-1"), "重置").Clicked() && m.OnReset != nil {
				m.OnReset(1)
			}
		})
		ResponseGraph(c, m)
		if m.AdvancedMode == 0 {
			GraphicEQ10(c, m)
		} else {
			GraphicEQ31(c, m)
		}
	})
}

func surroundPage(c *ui.Context, m *Model) {
	low := m.Pipeline.Surround.LowFreq * 100
	strength := m.Pipeline.Surround.Strength * 100
	ui.Column(c).Fill().Gap(12).Padding(8).Children(func() {
		panelHeader(c, m, "虚拟 3D 环绕", 3)
		ui.Row(c).Gap(12).AlignItems(ui.Center).Children(func() {
			if ui.Switch(c.Key("surround-on"), &m.Pipeline.Surround.Enabled).Label("启用环绕").Changed() {
				m.MarkDirty()
			}
			ui.Text(c, "通过虚拟声场处理，让普通耳机获得更宽的听感").FontSize(12).TextColor(colorMuted)
		})
		ui.Column(c).Gap(6).Padding(6, 0).Children(func() {
			Knob(c, m, "surround-low", "低频效果", &low, 0, 100, 1, "%", func() {
				m.Pipeline.Surround.LowFreq = low / 100
			})
			Knob(c, m, "surround-strength", "环绕强度", &strength, 0, 100, 1, "%", func() {
				m.Pipeline.Surround.Strength = strength / 100
			})
		})
		ui.Text(c, "建议从较低强度开始，避免过强的空间感造成疲劳。").FontSize(12).TextColor(colorMuted)
	})
}

func presetModal(c *ui.Context, m *Model) {
	ui.Modal(c, &m.PresetModalOpen, func() {
		ui.Column(c).Gap(12).Padding(20).Width(360).Background(colorSurface2).Border(1, colorBorder).Radius(10).Children(func() {
			ui.Text(c, "保存为预设").Bold().FontSize(16)
			ui.TextInput(c.Key("preset-name"), &m.PresetName).Label("预设名称").Placeholder("输入预设名称")
			ui.Row(c).Gap(8).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("preset-cancel"), "取消").Clicked() {
					m.PresetModalOpen = false
				}
				if ui.PrimaryButton(c.Key("preset-save"), "确定保存").Clicked() {
					if m.OnSavePreset != nil {
						m.OnSavePreset(m.PresetName)
					}
					m.PresetModalOpen = false
				}
			})
		})
	})
}

func balanceLabel(v float64) string {
	if v < -0.05 {
		return fmt.Sprintf("左 %.0f%%", -v*100)
	}
	if v > 0.05 {
		return fmt.Sprintf("右 %.0f%%", v*100)
	}
	return "居中"
}

// FormatDuration is kept in the ui package so the view does not depend on main.
func FormatDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d.Seconds())
	h := total / 3600
	min := (total % 3600) / 60
	sec := total % 60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, min, sec)
	}
	return fmt.Sprintf("%d:%02d", min, sec)
}
