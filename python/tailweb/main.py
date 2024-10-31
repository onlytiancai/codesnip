import time
import asyncio
import threading
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

async def stream_log(request: Request):
    start_time = time.time() 
    timeout_duration = 600 

    with open("log.txt", "r") as f:
        last_lines = deque(f, maxlen=100)
        for line in last_lines:
            yield f"data: {line}\n\n"

        while True:
            if time.time() - start_time > timeout_duration:
                print("Stream timeout.")
                break
            if await request.is_disconnected():
                print("Client disconnected")
                break
            
            try:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    print("Stream sleep:", time.time())
                    await asyncio.sleep(1)

            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                print("Stream connection closed by client")
                break


@app.get("/stream")
async def stream_endpoint(request: Request):
    return StreamingResponse(stream_log(request), media_type="text/event-stream")

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(request, 'index.html')

def write_time_to_log():
    with open("log.txt", "a") as f:
        while True:
            f.write(f"{datetime.now()}\n")
            f.flush()
            time.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=write_time_to_log, daemon=True)
    thread.start()
    yield

app.router.lifespan_context = lifespan