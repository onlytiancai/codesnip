from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse,PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import edge_tts

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )

async def tts(text, voice='zh-TW-HsiaoChenNeural'):
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]
        elif chunk["type"] == "WordBoundary":
            print(f"WordBoundary: {chunk}")

@app.get("/get_lines")
def get_lines(n: int, m: int) -> str:
    if n < 0:
        n = 0
    if m - n > 100:
        m = n + 100
    lines = []
    with open('/home/ubuntu/temp/yitian.txt', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            if n <= i <= m:
                line = line.strip()
                if line:
                    lines.append(line)
            elif i > m:
                break
    return PlainTextResponse('\n'.join(lines))

@app.get("/stream-mp3")
def stream_mp3(txt: str):
    return StreamingResponse(tts(txt[:512]), media_type="audio/mpeg")
