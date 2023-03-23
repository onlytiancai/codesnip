# gunicorn -w2 test:my_web_app --bind localhost:8080 --worker-class aiohttp.GunicornWebWorker
# ab -n 1000 -c 4 http://127.0.0.1:8080/
# https://y.tsutsumi.io/2017/09/23/aiohttp-vs-multithreaded-flask-for-high-io-applications/

from aiohttp import web
import aiohttp
import asyncio
import aiosqlite

db_name = 'test.db'

async def handle(request):
    async with aiohttp.ClientSession() as session:
        async with session.get('http://127.0.0.1') as response:
            html = await response.text()

    async with aiosqlite.connect(db_name) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT count(*) count FROM mytable") as cursor:
            row = await cursor.fetchone()

    name = request.match_info.get('name', "Anonymous")
    row_count = row['count']
    html_len = len(html)
    text = f'Hello, {name}, {row_count}, {html_len}\n'
    return web.Response(text=text)

async def init_db():
    async with aiosqlite.connect(db_name) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS mytable(id)")
        await db.execute("insert into mytable(id) values(?)", '1')
        await db.commit()

async def my_web_app():
    await init_db()
    app = web.Application()
    app.add_routes([web.get('/', handle),
                    web.get('/{name}', handle)])
    return app

if __name__ == '__main__':
    app = my_web_app()
    web.run_app(app)
