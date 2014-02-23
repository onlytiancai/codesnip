#!/usr/bin/env python

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Linux platform implementation."""

from __future__ import division

import os
import re
import warnings

from _common import *
from _compat import namedtuple

def get_system_boot_time():
    """Return the system boot time expressed in seconds since the epoch."""
    f = open('/proc/stat', 'r')
    try:
        for line in f:
            if line.startswith('btime'):
                return float(line.strip().split()[1])
        raise RuntimeError("line 'btime' not found")
    finally:
        f.close()

def get_num_cpus():
    """Return the number of CPUs on the system"""
    try:
        return os.sysconf("SC_NPROCESSORS_ONLN")
    except ValueError:
        # as a second fallback we try to parse /proc/cpuinfo
        num = 0
        f = open('/proc/cpuinfo', 'r')
        try:
            lines = f.readlines()
        finally:
            f.close()
        for line in lines:
            if line.lower().startswith('processor'):
                num += 1

    # unknown format (e.g. amrel/sparc architectures), see:
    # http://code.google.com/p/psutil/issues/detail?id=200
    # try to parse /proc/stat as a last resort
    if num == 0:
        f = open('/proc/stat', 'r')
        try:
            lines = f.readlines()
        finally:
            f.close()
        search = re.compile('cpu\d')
        for line in lines:
            line = line.split(' ')[0]
            if search.match(line):
                num += 1

    if num == 0:
        raise RuntimeError("couldn't determine platform's NUM_CPUS")
    return num


# Number of clock ticks per second
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
_PAGESIZE = os.sysconf("SC_PAGE_SIZE")

# Since these constants get determined at import time we do not want to
# crash immediately; instead we'll set them to None and most likely
# we'll crash later as they're used for determining process CPU stats
# and creation_time
try:
    BOOT_TIME = get_system_boot_time()
except Exception:
    BOOT_TIME = None
    warnings.warn("couldn't determine platform's BOOT_TIME", RuntimeWarning)
try:
    NUM_CPUS = get_num_cpus()
except Exception:
    NUM_CPUS = None
    warnings.warn("couldn't determine platform's NUM_CPUS", RuntimeWarning)

# ioprio_* constants http://linux.die.net/man/2/ioprio_get
IOPRIO_CLASS_NONE = 0
IOPRIO_CLASS_RT = 1
IOPRIO_CLASS_BE = 2
IOPRIO_CLASS_IDLE = 3

# --- system memory functions

nt_virtmem_info = namedtuple('vmem', ' '.join([
    # all platforms
    'total', 'available', 'percent', 'used', 'free',
    # linux specific
    'active',
    'inactive',
    'buffers',
    'cached']))

def virtual_memory():
    total, free, buffers, shared, _, _ = _psutil_linux.get_sysinfo()
    cached = active = inactive = None
    f = open('/proc/meminfo', 'r')
    try:
        for line in f:
            if line.startswith('Cached:'):
                cached = int(line.split()[1]) * 1024
            elif line.startswith('Active:'):
                active = int(line.split()[1]) * 1024
            elif line.startswith('Inactive:'):
                inactive = int(line.split()[1]) * 1024
            if cached is not None \
            and active is not None \
            and inactive is not None:
                break
        else:
            # we might get here when dealing with exotic Linux flavors, see:
            # http://code.google.com/p/psutil/issues/detail?id=313
            msg = "'cached', 'active' and 'inactive' memory stats couldn't " \
                  "be determined and were set to 0"
            warnings.warn(msg, RuntimeWarning)
            cached = active = inactive = 0
    finally:
        f.close()
    avail = free + buffers + cached
    used = total - free
    percent = usage_percent((total - avail), total, _round=1)
    return nt_virtmem_info(total, avail, percent, used, free,
                           active, inactive, buffers, cached)

def swap_memory():
    used = total - free
    percent = usage_percent(used, total, _round=1)
    # get pgin/pgouts
    f = open("/proc/vmstat", "r")
    sin = sout = None
    try:
        for line in f:
            # values are expressed in 4 kilo bytes, we want bytes instead
            if line.startswith('pswpin'):
                sin = int(line.split(' ')[1]) * 4 * 1024
            elif line.startswith('pswpout'):
                sout = int(line.split(' ')[1])  * 4 * 1024
            if sin is not None and sout is not None:
                break
        else:
            # we might get here when dealing with exotic Linux flavors, see:
            # http://code.google.com/p/psutil/issues/detail?id=313
            msg = "'sin' and 'sout' swap memory stats couldn't " \
                  "be determined and were set to 0"
            warnings.warn(msg, RuntimeWarning)
            sin = sout = 0
    finally:
        f.close()
    return nt_swapmeminfo(total, used, free, percent, sin, sout)

# --- XXX deprecated memory functions

@deprecated('psutil.virtual_memory().cached')
def cached_phymem():
    return virtual_memory().cached

@deprecated('psutil.virtual_memory().buffers')
def phymem_buffers():
    return virtual_memory().buffers


# --- system CPU functions

@memoize
def _get_cputimes_ntuple():
    """ Return a (nt, rindex) tuple depending on the CPU times available
    on this Linux kernel version which may be:
    user, nice, system, idle, iowait, irq, softirq [steal, [guest, [guest_nice]]]
    """
    f = open('/proc/stat', 'r')
    try:
        values = f.readline().split()[1:]
    finally:
        f.close()
    fields = ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq']
    rindex = 8
    vlen = len(values)
    if vlen >= 8:
        # Linux >= 2.6.11
        fields.append('steal')
        rindex += 1
    if vlen >= 9:
        # Linux >= 2.6.24
        fields.append('guest')
        rindex += 1
    if vlen >= 10:
        # Linux >= 3.2.0
        fields.append('guest_nice')
        rindex += 1
    return (namedtuple('cputimes', ' '.join(fields)), rindex)

def get_system_cpu_times():
    """Return a named tuple representing the following system-wide
    CPU times:
    user, nice, system, idle, iowait, irq, softirq [steal, [guest, [guest_nice]]]
    Last 3 fields may not be available on all Linux kernel versions.
    """
    f = open('/proc/stat', 'r')
    try:
        values = f.readline().split()
    finally:
        f.close()
    nt, rindex = _get_cputimes_ntuple()
    fields = values[1:rindex]
    fields = [float(x) / _CLOCK_TICKS for x in fields]
    return nt(*fields)

def get_system_per_cpu_times():
    """Return a list of namedtuple representing the CPU times
    for every CPU available on the system.
    """
    nt, rindex = _get_cputimes_ntuple()
    cpus = []
    f = open('/proc/stat', 'r')
    try:
        # get rid of the first line which refers to system wide CPU stats
        f.readline()
        for line in f:
            if line.startswith('cpu'):
                fields = line.split()[1:rindex]
                fields = [float(x) / _CLOCK_TICKS for x in fields]
                entry = nt(*fields)
                cpus.append(entry)
        return cpus
    finally:
        f.close()


# --- system disk functions

def disk_partitions(all=False):
    """Return mounted disk partitions as a list of nameduples"""
    phydevs = []
    f = open("/proc/filesystems", "r")
    try:
        for line in f:
            if not line.startswith("nodev"):
                phydevs.append(line.strip())
    finally:
        f.close()

    retlist = []
    for partition in partitions:
        device, mountpoint, fstype, opts = partition
        if device == 'none':
            device = ''
        if not all:
            if device == '' or fstype not in phydevs:
                continue
        ntuple = nt_partition(device, mountpoint, fstype, opts)
        retlist.append(ntuple)
    return retlist


# --- process functions

def get_pid_list():
    """Returns a list of PIDs currently running on the system."""
    pids = [int(x) for x in os.listdir('/proc') if x.isdigit()]
    return pids

def pid_exists(pid):
    """Check For the existence of a unix pid."""
    return _psposix.pid_exists(pid)

def net_io_counters():
    """Return network I/O statistics for every network interface
    installed on the system as a dict of raw tuples.
    """
    f = open("/proc/net/dev", "r")
    try:
        lines = f.readlines()
    finally:
        f.close()

    retdict = {}
    for line in lines[2:]:
        colon = line.rfind(':')
        assert colon > 0, repr(line)
        name = line[:colon].strip()
        fields = line[colon+1:].strip().split()
        bytes_recv = int(fields[0])
        packets_recv = int(fields[1])
        errin = int(fields[2])
        dropin = int(fields[3])
        bytes_sent = int(fields[8])
        packets_sent = int(fields[9])
        errout = int(fields[10])
        dropout = int(fields[11])
        retdict[name] = (bytes_sent, bytes_recv, packets_sent, packets_recv,
                         errin, errout, dropin, dropout)
    return retdict

def disk_io_counters():
    """Return disk I/O statistics for every disk installed on the
    system as a dict of raw tuples.
    """
    # man iostat states that sectors are equivalent with blocks and
    # have a size of 512 bytes since 2.4 kernels. This value is
    # needed to calculate the amount of disk I/O in bytes.
    SECTOR_SIZE = 512

    # determine partitions we want to look for
    partitions = []
    f = open("/proc/partitions", "r")
    try:
        lines = f.readlines()[2:]
    finally:
        f.close()
    for line in reversed(lines):
        _, _, _, name = line.split()
        if name[-1].isdigit():
            # we're dealing with a partition (e.g. 'sda1'); 'sda' will
            # also be around but we want to omit it
            partitions.append(name)
        else:
            if not partitions or not partitions[-1].startswith(name):
                # we're dealing with a disk entity for which no
                # partitions have been defined (e.g. 'sda' but
                # 'sda1' was not around), see:
                # http://code.google.com/p/psutil/issues/detail?id=338
                partitions.append(name)
    #
    retdict = {}
    f = open("/proc/diskstats", "r")
    try:
        lines = f.readlines()
    finally:
        f.close()
    for line in lines:
        # http://www.mjmwired.net/kernel/Documentation/iostats.txt
        _, _, name, reads, _, rbytes, rtime, writes, _, wbytes, wtime = \
            line.split()[:11]
        if name in partitions:
            rbytes = int(rbytes) * SECTOR_SIZE
            wbytes = int(wbytes) * SECTOR_SIZE
            reads = int(reads)
            writes = int(writes)
            rtime = int(rtime)
            wtime = int(wtime)
            retdict[name] = (reads, writes, rbytes, wbytes, rtime, wtime)
    return retdict
