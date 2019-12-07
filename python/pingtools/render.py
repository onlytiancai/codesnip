#-*- coding=utf-8 -*-
import os
import cStringIO
import tempfile
import datetime

import rrdtool
from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image

def wn_status(request):
    rrd_name = os.path.join(BASE_DIR, "rrddatas/gw.rrd")
    fd, path = tempfile.mkstemp('.png')
    # 时间格式中的 : 也要转义
    dtime = datetime.datetime.strftime(
        datetime.datetime.now(),
        "%Y-%m-%d %H\:%M\:%S"
    )
    rrdtool.graph(path,
                  '--start', "-12h",
                  "-t", "渭南网关带宽",
                  "-w", "600",
                  "-h", "200",
                  'DEF:value1={0}:wan_rx:AVERAGE'.format(rrd_name),
                  'DEF:value2={0}:wan_tx:AVERAGE'.format(rrd_name),
                  'AREA:value1#00ff00: rx',
                  'LINE:value2#ff0000: tx',
                  "COMMENT:\\n",
                  'AREA:value1#00ff00:下载',
                  'GPRINT:value1:LAST:当前\:%8.0lf',
                  'GPRINT:value1:AVERAGE:平均\:%8.0lf',
                  'GPRINT:value1:MAX:最大\:%8.0lf',
                  'GPRINT:value1:MIN:最小\:%8.0lf',
                  "COMMENT:\\n",
                  'LINE2:value2#4433ff:上传',
                  'GPRINT:value2:LAST:当前\:%8.0lf',
                  'GPRINT:value2:AVERAGE:平均\:%8.0lf',
                  'GPRINT:value2:MAX:最大\:%8.0lf',
                  'GPRINT:value2:MIN:最小\:%8.0lf',
                  "COMMENT:\\n",
                  "COMMENT:Date " + dtime)

    im = Image.open(path)
    out = cStringIO.StringIO()
    im.save(out, format='png')
    room = out.getvalue()
    out.close()
    os.remove(path)
    return HttpResponse(room, 'image/png')
