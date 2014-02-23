# -*- coding: utf-8 -*-
'''
git://github.com/justquick/google-chartwrapper.git
'''
from gchart import *
G = Pie3D([91, 9], encoding='text')
G.size(500, 200)
G.label('正常', '宕机')
G.title('www.dnspod.cn可用率概要')
G.color('4CBB47','FF9900')
print G.url

G = HorizontalBarGroup( [[100],[80],[90]], encoding='text' )
G.size(300,130)
G.color('FF9900','FFCC33', 'cc28dd')
G.axes('xy')
G.axes.label(1, None,'20','40','60','80', '100')
G.marker('t100%','black',0,0,13)
G.marker('t80%','black',1,0,13)
G.marker('t90%','black',2,0,13)
G.title('www.dnspod.cn可用率详情')
G.legend('10.133.42.139','72.90.87.64','251.32.79.90')
print G.url
