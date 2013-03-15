# -*- coding: utf-8 -*-
import redis
from datetime import datetime

r = redis.StrictRedis()

lua = """
local value = redis.call("HGET",KEYS[1],KEYS[2])
value = tonumber(value)
local new_value = tonumber(ARGV[1])
if new_value > value then
    redis.call("HSET", KEYS[1], KEYS[2], new_value)
end
"""
set_max = r.register_script(lua)
begin_time = datetime.now()

for i in range(10000):
    set_max(['test', 'c'], [i])

print r.hget('test', 'c')
end_time = datetime.now()
print end_time - begin_time
