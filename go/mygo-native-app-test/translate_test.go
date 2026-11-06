package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// testClient returns a client that talks to the given server.
func testClient(host string) *ollamaClient {
	return &ollamaClient{
		host:  host,
		model: "test/model",
		client: &http.Client{
			Transport: &http.Transport{Proxy: nil},
		},
	}
}

// TestTranslate sends the request Ollama expects and reads what it
// answers. The prompt has to name both languages: without it the model
// treats the text as a conversation instead of something to translate.
func TestTranslate(t *testing.T) {
	var gotPath, gotMethod string
	var got generateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotMethod = r.URL.Path, r.Method
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &got); err != nil {
			t.Errorf("请求体不是 JSON：%s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"response":"  I like artificial intelligence.  "}`)
	}))
	defer srv.Close()

	out, err := testClient(srv.URL).translate("中文", "English", "我喜欢人工智能。")
	if err != nil {
		t.Fatal(err)
	}
	if out != "I like artificial intelligence." {
		t.Errorf("译文 %q", out)
	}
	if gotMethod != http.MethodPost {
		t.Errorf("方法 %s", gotMethod)
	}
	if gotPath != "/api/generate" {
		t.Errorf("路径 %s", gotPath)
	}
	if got.Model != "test/model" {
		t.Errorf("模型 %q", got.Model)
	}
	if want := "Translate from Chinese to English: 我喜欢人工智能。"; got.Prompt != want {
		t.Errorf("提示 %q，应为 %q", got.Prompt, want)
	}
}

// TestTranslatePromptNames checks that every language reaches the model
// under the name it knows, not under its label.
func TestTranslatePromptNames(t *testing.T) {
	cases := []struct {
		from, to, want string
	}{
		{"中文", "English", "Translate from Chinese to English: "},
		{"中文", "日本語", "Translate from Chinese to Japanese: "},
		{"English", "Français", "Translate from English to French: "},
	}
	for _, c := range cases {
		var got generateRequest
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &got)
			io.WriteString(w, `{"response":"ok"}`)
		}))
		if _, err := testClient(srv.URL).translate(c.from, c.to, "文本"); err != nil {
			srv.Close()
			t.Fatal(err)
		}
		srv.Close()
		if !strings.HasPrefix(got.Prompt, c.want) {
			t.Errorf("%s→%s 的提示 %q，应以 %q 开头", c.from, c.to, got.Prompt, c.want)
		}
	}
}

// TestTranslateErrors checks the failures the interface shows.
func TestTranslateErrors(t *testing.T) {
	t.Run("模型报错", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, `{"error":"model not found"}`)
		}))
		defer srv.Close()
		_, err := testClient(srv.URL).translate("中文", "English", "你好")
		if err == nil || !strings.Contains(err.Error(), "model not found") {
			t.Errorf("错误 %v", err)
		}
	})
	t.Run("服务器报错", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, "something broke")
		}))
		defer srv.Close()
		_, err := testClient(srv.URL).translate("中文", "English", "你好")
		if err == nil || !strings.Contains(err.Error(), "500") {
			t.Errorf("错误 %v", err)
		}
	})
	t.Run("没有译文", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, `{"response":"  "}`)
		}))
		defer srv.Close()
		_, err := testClient(srv.URL).translate("中文", "English", "你好")
		if err == nil {
			t.Error("空译文应报错")
		}
	})
	t.Run("连不上服务器", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		host := srv.URL
		srv.Close() // nothing listens there now
		_, err := testClient(host).translate("中文", "English", "你好")
		if err == nil || !strings.Contains(err.Error(), "Ollama") {
			t.Errorf("错误 %v，应说明连不上 Ollama", err)
		}
	})
}

// TestLangName checks the name the model knows a label by.
func TestLangName(t *testing.T) {
	if got := langName("中文"); got != "Chinese" {
		t.Errorf("中文 的名字 %q", got)
	}
	if got := langName("Español"); got != "Spanish" {
		t.Errorf("Español 的名字 %q", got)
	}
	if got := langName("Klingon"); got != "Klingon" {
		t.Errorf("没见过的语言 %q", got)
	}
}

// TestLangLabels checks that every language has a label the interface can
// show, and that the labels are the ones it knows.
func TestLangLabels(t *testing.T) {
	if len(langLabels) != len(langs) {
		t.Fatalf("%d 个标签，%d 种语言", len(langLabels), len(langs))
	}
	seen := map[string]bool{}
	for i, l := range langs {
		if langLabels[i] != l.label {
			t.Errorf("标签 %d 是 %q，应为 %q", i, langLabels[i], l.label)
		}
		if seen[l.label] {
			t.Errorf("标签 %q 重复", l.label)
		}
		seen[l.label] = true
		if langName(l.label) != l.name {
			t.Errorf("标签 %q 的名字 %q", l.label, langName(l.label))
		}
	}
}
