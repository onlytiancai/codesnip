package dsp

import (
	"io"
	"math"
	"testing"
)

func sineRMS(b *Biquad, freq, sampleRate float64, seconds float64) float64 {
	total := int(seconds * sampleRate)
	warm := int(0.1 * sampleRate)
	sum := 0.0
	count := 0
	for i := 0; i < total; i++ {
		x := math.Sin(2 * math.Pi * freq * float64(i) / sampleRate)
		y := b.Process(x)
		if i >= warm {
			sum += y * y
			count++
		}
	}
	return math.Sqrt(sum / float64(count))
}

func plainRMS(freq, sampleRate float64, seconds float64) float64 {
	return sineRMS(NewBiquad(Coefficients{B0: 1}), freq, sampleRate, seconds)
}

func TestBiquadPeakingBoost(t *testing.T) {
	sr := 48000.0
	base := plainRMS(1000, sr, 0.5)
	b := NewBiquad(DesignCoefficients(Peaking, 1000, 6, 1, sr))
	got := sineRMS(b, 1000, sr, 0.5)
	ratio := got / base
	if ratio < 1.7 || ratio > 2.2 {
		t.Fatalf("1 kHz +6 dB ratio = %.3f, want about 2", ratio)
	}
}

func TestBiquadLowpassAttenuatesHighFrequencies(t *testing.T) {
	sr := 48000.0
	base := plainRMS(8000, sr, 0.3)
	b := NewBiquad(DesignCoefficients(Lowpass, 1000, 0, 0.707, sr))
	got := sineRMS(b, 8000, sr, 0.3)
	if got/base > 0.3 {
		t.Fatalf("8 kHz after 1 kHz lowpass = %.4f, want much less than %.4f", got, base)
	}
}

func TestFilterChainChannelsAreIndependent(t *testing.T) {
	p := DefaultPipeline()
	p.Basic.Bass = 6
	chain := NewFilterChain(p, 48000)
	l := make([]float64, 4800)
	r := make([]float64, 4800)
	for i := range l {
		l[i] = math.Sin(2 * math.Pi * 100 * float64(i) / 48000)
	}
	chain.ProcessBlock(l, r)
	for i, v := range r {
		if math.Abs(v) > 1e-9 {
			t.Fatalf("right channel leaked at %d: %v", i, v)
		}
	}
	if chain.Count() == 0 {
		t.Fatal("expected one active filter")
	}
}

func TestResponseMatchesPeakingGain(t *testing.T) {
	p := DefaultPipeline()
	p.Basic.Mid = 6
	got := Response(p, 48000, []float64{1000})[0]
	if got < 5.5 || got > 6.5 {
		t.Fatalf("response at 1 kHz = %.2f dB, want about 6", got)
	}
}

func TestBandwidthQRoundTrip(t *testing.T) {
	q := 1.7
	bw := QToBandwidth(q, 1000, 48000)
	got := BandwidthToQ(bw, 1000, 48000)
	if math.Abs(got-q) > 0.05 {
		t.Fatalf("Q round trip = %.3f, want %.3f", got, q)
	}
}

func TestBalanceAndLimiter(t *testing.T) {
	l, r := ApplyBalance(1, 1, 0, true, true)
	if math.Abs(l-1) > 1e-9 || math.Abs(r-1) > 1e-9 {
		t.Fatalf("center balance = %.3f, %.3f", l, r)
	}
	l, r = ApplyBalance(1, 1, -1, true, true)
	if math.Abs(l-math.Sqrt2) > 1e-9 || math.Abs(r) > 1e-9 {
		t.Fatalf("hard left = %.3f, %.3f", l, r)
	}

	lim := NewLimiter(48000, true)
	for i := 0; i < 2000; i++ {
		l, r = lim.Process(4, -4)
		if math.Abs(l) > 1.001 || math.Abs(r) > 1.001 {
			t.Fatalf("limiter output = %.3f, %.3f", l, r)
		}
	}
}

func TestSpectrumFindsSine(t *testing.T) {
	s := NewSpectrum(48000)
	freq := 1500.0
	for i := 0; i < spectrumFFTSize; i++ {
		x := math.Sin(2 * math.Pi * freq * float64(i) / 48000)
		s.Push(x, x)
	}
	bars := s.Snapshot()
	best, bestIdx := float32(0), 0
	for i, v := range bars {
		if v > best {
			best, bestIdx = v, i
		}
	}
	want := int(float64(SpectrumBins-1) * math.Log(freq/20) / math.Log(1000))
	if math.Abs(float64(bestIdx-want)) > 8 {
		t.Fatalf("peak bar = %d, want near %d", bestIdx, want)
	}
}

func TestSurroundBypassAndFiniteOutput(t *testing.T) {
	s := NewSurroundProcessor()
	s.Update(Surround{}, 48000)
	l, r := []float64{1}, []float64{0.5}
	s.ProcessBlock(l, r)
	if l[0] != 1 || r[0] != 0.5 {
		t.Fatalf("bypass changed samples: %v %v", l, r)
	}

	s.Update(Surround{Enabled: true, Strength: 0.6, LowFreq: 0.4}, 48000)
	ll := make([]float64, 1000)
	rr := make([]float64, 1000)
	for i := range ll {
		ll[i] = math.Sin(float64(i) * 0.1)
		rr[i] = math.Cos(float64(i) * 0.07)
	}
	s.ProcessBlock(ll, rr)
	for i := range ll {
		if math.IsNaN(ll[i]) || math.IsNaN(rr[i]) || math.IsInf(ll[i], 0) || math.IsInf(rr[i], 0) {
			t.Fatalf("non-finite surround output at %d", i)
		}
	}
}

type fakeDecoder struct {
	pos int64
	sr  int
	ch  int
	len int64
}

func (f *fakeDecoder) SampleRate() int { return f.sr }
func (f *fakeDecoder) Channels() int   { return f.ch }
func (f *fakeDecoder) Len() int64      { return f.len }
func (f *fakeDecoder) SetPosition(frame int64) error {
	f.pos = frame
	return nil
}
func (f *fakeDecoder) Read(buf []float32) (int, error) {
	if f.pos >= f.len {
		return 0, io.EOF
	}
	n := int64(len(buf))
	if f.pos+n > f.len {
		n = f.len - f.pos
	}
	for i := int64(0); i < n; i++ {
		buf[i] = 0.5
	}
	f.pos += n
	return int(n), nil
}

func TestDecoderWrapsStereoAndMono(t *testing.T) {
	p := DefaultPipeline()
	st := NewDecoder(&fakeDecoder{sr: 48000, ch: 2, len: 100}, p)
	buf := make([]float32, 8)
	n, _ := st.Read(buf)
	if n != 8 {
		t.Fatalf("stereo read = %d, want 8", n)
	}
	for i, v := range buf {
		if math.IsNaN(float64(v)) || math.Abs(float64(v)) > 1 {
			t.Fatalf("stereo sample %d = %v", i, v)
		}
	}

	mono := NewDecoder(&fakeDecoder{sr: 48000, ch: 1, len: 100}, p)
	if mono.Channels() != 2 {
		t.Fatalf("wrapped channels = %d, want 2", mono.Channels())
	}
	n, _ = mono.Read(buf)
	if n != 8 || buf[0] != buf[1] {
		t.Fatalf("mono duplication failed: n=%d samples=%v", n, buf[:4])
	}
}
