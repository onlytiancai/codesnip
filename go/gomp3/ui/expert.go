package ui

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"

	"gomp3/dsp"
)

// Expert table column widths. The header and every band row use exactly the
// same widths so labels line up with their controls.
const (
	colEnable = 44
	colType   = 104
	colFreq   = 104
	colGain   = 108
	colQ      = 224
	colDelete = 66
)

// ExpertEQ edits an arbitrary number of parametric filter bands.
func ExpertEQ(c *ui.Context, m *Model) {
	ResponseGraph(c, m)

	ui.Row(c).Gap(8).AlignItems(ui.Center).Padding(6, 0).Children(func() {
		ui.Text(c, "参数均衡器").Bold().FontSize(15)
		ui.Text(c, fmt.Sprintf("%d 个波段", len(m.Pipeline.Params))).FontSize(12).TextColor(colorMuted)
		ui.Spacer(c)
		if ui.Button(c.Key("add-band"), "+ 添加波段").Clicked() {
			m.Pipeline.Params = append(m.Pipeline.Params, dsp.NewParametricBand())
			m.MarkDirty()
		}
		if ui.Button(c.Key("reset-2"), "重置").Clicked() && m.OnReset != nil {
			m.OnReset(2)
		}
	})

	ui.Scroll(c).Grow(1).Gap(6).Children(func() {
		// Header row: same widths as the data rows below.
		ui.Row(c).Gap(6).AlignItems(ui.Center).Padding(4, 6).Background(colorPanel).Radius(6).Children(func() {
			ui.Box(c).Width(colEnable).Children(func() {
				ui.Text(c, "启用").FontSize(11).TextColor(colorMuted)
			})
			ui.Box(c).Width(colType).Children(func() {
				ui.Text(c, "类型").FontSize(11).TextColor(colorMuted)
			})
			ui.Box(c).Width(colFreq).Children(func() {
				ui.Text(c, "频率 Hz").FontSize(11).TextColor(colorMuted)
			})
			ui.Box(c).Width(colGain).Children(func() {
				ui.Text(c, "增益 dB").FontSize(11).TextColor(colorMuted)
			})
			ui.Box(c).Width(colQ).Children(func() {
				ui.Text(c, "Q / 带宽").FontSize(11).TextColor(colorMuted)
			})
			ui.Box(c).Width(colDelete)
		})

		for i := range m.Pipeline.Params {
			band := &m.Pipeline.Params[i]
			key := band.ID
			if key == "" {
				key = fmt.Sprintf("band-%d", i)
			}
			ui.Row(c.Key(key)).Gap(6).AlignItems(ui.Center).Padding(5, 6).Background(colorSurface).Radius(6).Children(func() {
				// 启用
				ui.Box(c).Width(colEnable).Children(func() {
					if ui.Checkbox(c.Key(key+"-on"), &band.Enabled, "").Changed() {
						m.MarkDirty()
					}
				})
				// 类型
				ui.Box(c).Width(colType).Children(func() {
					typeLabel := band.Type.String()
					sel := ui.Select(c.Key(key+"-type"), &typeLabel, dsp.FilterTypes()).Label("滤波器类型")
					if sel.Changed() {
						band.Type = dsp.FilterTypeFromLabel(typeLabel)
						m.MarkDirty()
					}
				})
				// 频率
				ui.Box(c).Width(colFreq).Children(func() {
					freq := band.Freq
					nf := ui.NumberInput(c.Key(key+"-freq"), &freq, 20, 20000, 1).Label("频率")
					if nf.Changed() {
						band.Freq = dsp.Clamp(freq, 20, 20000)
						m.MarkDirty()
					}
				})
				// 增益
				ui.Box(c).Width(colGain).Children(func() {
					gain := band.GainDB
					ng := ui.NumberInput(c.Key(key+"-gain"), &gain, -24, 24, 0.5).Label("增益")
					if ng.Changed() {
						band.GainDB = dsp.Clamp(gain, -24, 24)
						m.MarkDirty()
					}
				})
				// Q / 带宽
				ui.Box(c).Width(colQ).Children(func() {
					ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
						if ui.Checkbox(c.Key(key+"-bw"), &band.UseBandwidth, "带宽").Changed() {
							if band.UseBandwidth {
								band.BandwidthOct = dsp.QToBandwidth(band.Q, band.Freq, 44100)
							} else {
								band.Q = dsp.BandwidthToQ(band.BandwidthOct, band.Freq, 44100)
							}
							m.MarkDirty()
						}
						if band.UseBandwidth {
							bw := band.BandwidthOct
							nb := ui.NumberInput(c.Key(key+"-bwval"), &bw, 0.1, 4, 0.1).Width(150).Label("带宽八度")
							if nb.Changed() {
								band.BandwidthOct = dsp.Clamp(bw, 0.1, 4)
								m.MarkDirty()
							}
						} else {
							q := band.Q
							nq := ui.NumberInput(c.Key(key+"-q"), &q, 0.1, 20, 0.1).Width(150).Label("Q 值")
							if nq.Changed() {
								band.Q = math.Max(0.1, math.Min(20, q))
								m.MarkDirty()
							}
						}
					})
				})
				// 删除
				ui.Box(c).Width(colDelete).Children(func() {
					if ui.Button(c.Key(key+"-del"), "删除").Clicked() {
						m.Pipeline.Params = append(m.Pipeline.Params[:i], m.Pipeline.Params[i+1:]...)
						m.MarkDirty()
						return
					}
				})
			})
		}
		if len(m.Pipeline.Params) == 0 {
			ui.Text(c, "还没有参数波段。点击“添加波段”开始精细调整。").FontSize(12).TextColor(colorMuted).Padding(12, 6)
		}
	})
}
