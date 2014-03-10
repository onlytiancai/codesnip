import statsd
c = statsd.StatsClient('172.4.0.23', 8125)
for i in range(20):
    c.incr('huhao.foo')
print 'ok'
