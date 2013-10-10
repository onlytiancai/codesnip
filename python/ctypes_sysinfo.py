# -*- coding: utf-8 -*-
from ctypes import * # NOQA

'''
http://stackoverflow.com/questions/3449442/help-me-understand-why-my-trivial-use-of-pythons-ctypes-module-is-failing
struct sysinfo {
    long uptime;
    unsigned long loads[3];
    unsigned long totalram;
    unsigned long freeram;
    unsigned long sharedram;
    unsigned long bufferram;
    unsigned long totalswap;
    unsigned long freeswap;
    unsigned short procs;
    unsigned short pad;
    unsigned long totalhigh;
    unsigned long freehigh;
    unsigned int mem_unit;
    char _f[20-2*sizeof(long)-sizeof(int)];
};

'''


class sysinfo(Structure):
    _fields_ = [("uptime", c_long),
                ("loads", c_ulong * 3),
                ("totalram", c_ulong),
                ("freeram", c_ulong),
                ("sharedram", c_ulong),
                ("bufferram", c_ulong),
                ("totalswap", c_ulong),
                ("freeswap", c_ulong),
                ("procs", c_short),
                ("pad", c_short),
                ("totalhigh", c_ulong),
                ("freehigh", c_ulong),
                ("mem_unit", c_int),
                ("_f", c_char * (20 - 2 * sizeof(c_long) - sizeof(c_int))),
                ]

libc = CDLL('libc.so.6')
libc.printf('hello world.\n')

s_info = sysinfo()
error = libc.sysinfo(byref(s_info))
print "code error=%d" % error
print '''Uptime = %ds
Load: 1 min %d / 5 min %d / 15 min %d
RAM: total %d / free %d /shared %d
Memory in buffers = %d
Swap:total %d / free %d
Number of processes = %d''' \
  % (s_info.uptime, s_info.loads[0],
     s_info.loads[1], s_info.loads[2],
     s_info.totalram, s_info.freeram, s_info.sharedram, s_info.bufferram,
     s_info.totalswap, s_info.freeswap,
     s_info.procs)
