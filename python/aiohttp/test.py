# gunicorn -w2 test:my_web_app --bind localhost:8080 --worker-class aiohttp.GunicornWebWorker
# ab -n 1000 -c 4 http://127.0.0.1:8080/

from aiohttp import web
import aiohttp
import asyncio

async def handle(request):
    async with aiohttp.ClientSession() as session:
        async with session.get('http://127.0.0.1') as response:
            html = await response.text()
            name = request.match_info.get('name', "Anonymous")
            text = "Hello, " + name + '\n'
            return web.Response(text=text)

async def my_web_app():
    app = web.Application()
    app.add_routes([web.get('/', handle),
                    web.get('/{name}', handle)])
    return app

if __name__ == '__main__':
    app = my_web_app()
    web.run_app(app)
