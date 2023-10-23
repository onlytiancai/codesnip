from flask import Flask
from flask.ext.aiohttp import async, AioHTTP

app = Flask(__name__)
aio = AioHTTP(app)

@app.route('/use-external-api')
@async
def use_external_api():
    response = yield from aiohttp.request(
        'GET', 'http://127.0.0.1')
    data = yield from response.read()

    return data

if __name__ == '__main__':
    aio.run(app, debug=True)
