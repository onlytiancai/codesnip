#!/usr/bin/env python3
"""Drive the built app through MyGo's automation plugin.

The app has to be built with the plugin and run with a socket to talk to:

    go build -tags e2e -o /tmp/mygo-e2e .
    MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e &
    python3 e2e/drive.py

It reads the window's elements the way a screen reader does, types into
them, presses the buttons, waits for the model to answer and writes
pictures of the window to /tmp.

The window has to be on screen: a window behind another one is not drawn,
and the elements the app reports are those of the last frame it drew.
So the driver brings the app to the front first. Set MYGO_APP to the name
of the process when it is not "mygo-e2e".

The protocol is newline-delimited JSON over a Unix socket; see
<https://github.com/egoist/mygo/blob/main/docs/plugins/automation.md>.
"""
import base64
import json
import os
import socket
import subprocess
import sys
import time

SOCK = os.environ.get("MYGO_AUTOMATION", "/tmp/mygo-e2e.sock")
APP = os.environ.get("MYGO_APP", "mygo-e2e")
PREFIX = "/tmp/app"


class Client:
    """One connection to the app. Requests are answered in order."""

    def __init__(self, path=SOCK):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.f = self.sock.makefile("rwb")
        self.id = 0

    def call(self, method, params=None):
        self.id += 1
        req = {"id": self.id, "method": method}
        if params:
            req["params"] = params
        self.f.write((json.dumps(req) + "\n").encode())
        self.f.flush()
        ans = json.loads(self.f.readline())
        if "error" in ans:
            raise SystemExit(f"{method} failed: {ans['error']}")
        return ans["result"]


c = Client()
win = c.call("windows")["windows"][0]["id"]

# Bring the app to the front: a window behind another one is not drawn,
# and the elements the app reports are those of the last frame it drew.
subprocess.run(
    ["osascript", "-e",
     f'tell application "System Events" to set frontmost of process "{APP}" to true'],
    check=False,
)
time.sleep(1)


def state(refresh=False):
    """The value and the states of every element that has a key.

    The elements are those of the last frame the app drew, and a window
    behind another one is not drawn. So ask for a capture first: it draws
    the frame, whether the window is on screen or not.
    """
    if refresh:
        c.call("capture", {"window": win})
    out = {}
    for n in c.call("tree", {"window": win})["nodes"]:
        if n.get("key"):
            out[n["key"]] = (n.get("value"), n.get("states"))
    return out


def act(key, action, text=None):
    params = {"window": win, "key": key, "action": action}
    if text is not None:
        params["text"] = text
    c.call("act", params)


def wait_for_result(before=None, timeout=60):
    """The translation, once the model has answered and the frame is drawn.

    The result of the last translation is still on screen, so wait for one
    that is not it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.4)
        value = state(refresh=True).get("result", (None,))[0]
        if value and value != before:
            return value
    raise SystemExit(f"the model did not answer within {timeout}s")


def shot(name):
    png = c.call("capture", {"window": win})
    path = f"{PREFIX}-{name}.png"
    with open(path, "wb") as f:
        f.write(base64.b64decode(png["png"]))
    print(f"  wrote {path} ({png['width']}x{png['height']})")


def direction():
    s = state()
    return s["from"][0], s["to"][0]


def translate(text):
    """Translate text and return what came back."""
    before = state().get("result", (None,))[0]
    act("source", "setValue", text)
    act("translate", "press")
    return wait_for_result(before)


# The app keeps the languages the user left it with, so put them back.
if direction()[0] != "中文":
    act("swap", "press")
    time.sleep(0.5)
print("direction:", " -> ".join(direction()))

# Chinese to English.
print("zh -> en:", translate("我喜欢人工智能。今天天气很好。"))
shot("zh-en")

# The copy button puts the translation on the clipboard.
act("copy", "press")
time.sleep(0.5)
clip = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
print("clipboard:", repr(clip))

# Swap the languages and translate the other way.
act("swap", "press")
time.sleep(0.5)
print("after swap:", " -> ".join(direction()))
print("en -> zh:", translate("I like artificial intelligence."))
shot("en-zh")
