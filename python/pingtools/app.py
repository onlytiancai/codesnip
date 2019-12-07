import os
import tempfile
import datetime

import rrdtool
from settings import BASE_DIR, addrs, names

from flask import Flask, send_file, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', addrs=addrs, names=names)

@app.route('/show')
def show():
    name = request.args['name']
    ip = request.args['ip']
    rrd_name = os.path.join(BASE_DIR, "rrddatas/%s.rrd" % ip)
    fd, path = tempfile.mkstemp('.png')
    # 时间格式中的 : 也要转义
    dtime = datetime.datetime.strftime(
        datetime.datetime.now(),
        "%Y-%m-%d %H\:%M\:%S"
    )
    rrdtool.graph(path,
                  '--start', "-1h",
                  "-t", name + ' ' + ip,
                  "-w", "400",
                  "-h", "200",
                  'DEF:value1={0}:rtt:AVERAGE'.format(rrd_name),
                  'AREA:value1#00ff00:rtt',
                  'GPRINT:value1:LAST:last\:%8.0lf ms',
                  'GPRINT:value1:AVERAGE:avg\:%8.0lf ms',
                  'GPRINT:value1:MAX:max\:%8.0lf ms',
                  'GPRINT:value1:MIN:min\:%8.0lf ms',
                  "COMMENT:Date " + dtime)

    return send_file(path, attachment_filename='logo.png', mimetype='image/png')
