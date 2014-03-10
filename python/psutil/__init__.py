#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""psutil is a module providing convenience functions for managing
processes and gather system information in a portable way by using
Python.
"""

import os
import time
from collections import namedtuple

from _common import (nt_disk_iostat as _nt_disk_iostat,
                     nt_net_iostat as _nt_net_iostat,
                     )


import psutil._pslinux as _psplatform

NUM_CPUS = _psplatform.NUM_CPUS
BOOT_TIME = _psplatform.BOOT_TIME


# =====================================================================
# --- CPU related functions
# =====================================================================

def cpu_times(percpu=False):
    """Return system-wide CPU times as a namedtuple object.
    Every CPU time represents the time CPU has spent in the given mode.
    The attributes availability varies depending on the platform.
    Here follows a list of all available attributes:
     - user
     - system
     - idle
     - nice (UNIX)
     - iowait (Linux)
     - irq (Linux, FreeBSD)
     - softirq (Linux)
     - steal (Linux >= 2.6.11)
     - guest (Linux >= 2.6.24)
     - guest_nice (Linux >= 3.2.0)

    When percpu is True return a list of nameduples for each CPU.
    First element of the list refers to first CPU, second element
    to second CPU and so on.
    The order of the list is consistent across calls.
    """
    if not percpu:
        return _psplatform.get_system_cpu_times()
    else:
        return _psplatform.get_system_per_cpu_times()


_last_cpu_times = cpu_times()
_last_per_cpu_times = cpu_times(percpu=True)


def cpu_percent(interval=0.1, percpu=False):
    """Return a float representing the current system-wide CPU
    utilization as a percentage.

    When interval is > 0.0 compares system CPU times elapsed before
    and after the interval (blocking).

    When interval is 0.0 or None compares system CPU times elapsed
    since last call or module import, returning immediately.
    In this case is recommended for accuracy that this function be
    called with at least 0.1 seconds between calls.

    When percpu is True returns a list of floats representing the
    utilization as a percentage for each CPU.
    First element of the list refers to first CPU, second element
    to second CPU and so on.
    The order of the list is consistent across calls.

    Examples:

      >>> # blocking, system-wide
      >>> psutil.cpu_percent(interval=1)
      2.0
      >>>
      >>> # blocking, per-cpu
      >>> psutil.cpu_percent(interval=1, percpu=True)
      [2.0, 1.0]
      >>>
      >>> # non-blocking (percentage since last call)
      >>> psutil.cpu_percent(interval=0)
      2.9
      >>>
    """
    global _last_cpu_times
    global _last_per_cpu_times
    blocking = interval is not None and interval > 0.0

    def calculate(t1, t2):
        t1_all = sum(t1)
        t1_busy = t1_all - t1.idle

        t2_all = sum(t2)
        t2_busy = t2_all - t2.idle

        # this usually indicates a float precision issue
        if t2_busy <= t1_busy:
            return 0.0

        busy_delta = t2_busy - t1_busy
        all_delta = t2_all - t1_all
        busy_perc = (busy_delta / all_delta) * 100
        return round(busy_perc, 1)

    # system-wide usage
    if not percpu:
        if blocking:
            t1 = cpu_times()
            time.sleep(interval)
        else:
            t1 = _last_cpu_times
        _last_cpu_times = cpu_times()
        return calculate(t1, _last_cpu_times)
    # per-cpu usage
    else:
        ret = []
        if blocking:
            tot1 = cpu_times(percpu=True)
            time.sleep(interval)
        else:
            tot1 = _last_per_cpu_times
        _last_per_cpu_times = cpu_times(percpu=True)
        for t1, t2 in zip(tot1, _last_per_cpu_times):
            ret.append(calculate(t1, t2))
        return ret


# Use separate global vars for cpu_times_percent() so that it's
# independent from cpu_percent() and they can both be used within
# the same program.
_last_cpu_times_2 = _last_cpu_times
_last_per_cpu_times_2 = _last_per_cpu_times
_ptime_cpu_perc_nt = None


def cpu_times_percent(interval=0.1, percpu=False):
    """Same as cpu_percent() but provides utilization percentages
    for each specific CPU time as is returned by cpu_times().
    For instance, on Linux we'll get:

      >>> cpu_times_percent()
      cpupercent(user=4.8, nice=0.0, system=4.8, idle=90.5, iowait=0.0,
                 irq=0.0, softirq=0.0, steal=0.0, guest=0.0, guest_nice=0.0)
      >>>

    interval and percpu arguments have the same meaning as in
    cpu_percent().
    """
    global _last_cpu_times_2
    global _last_per_cpu_times_2
    blocking = interval is not None and interval > 0.0
    WINDOWS = os.name == 'nt'

    def calculate(t1, t2):
        global _ptime_cpu_perc_nt
        nums = []
        all_delta = sum(t2) - sum(t1)
        for field in t1._fields:
            field_delta = getattr(t2, field) - getattr(t1, field)
            try:
                field_perc = (100 * field_delta) / all_delta
            except ZeroDivisionError:
                field_perc = 0.0
            field_perc = round(field_perc, 1)
            if WINDOWS:
                # XXX
                # Work around:
                # https://code.google.com/p/psutil/issues/detail?id=392
                # CPU times are always supposed to increase over time
                # or at least remain the same and that's because time
                # cannot go backwards.
                # Surprisingly sometimes this might not be the case on
                # Windows where 'system' CPU time can be smaller
                # compared to the previous call, resulting in corrupted
                # percentages (< 0 or > 100).
                # I really don't know what to do about that except
                # forcing the value to 0 or 100.
                if field_perc > 100.0:
                    field_perc = 100.0
                elif field_perc < 0.0:
                    field_perc = 0.0
            nums.append(field_perc)
        if _ptime_cpu_perc_nt is None:
            _ptime_cpu_perc_nt = namedtuple('cpupercent', ' '.join(t1._fields))
        return _ptime_cpu_perc_nt(*nums)

    # system-wide usage
    if not percpu:
        if blocking:
            t1 = cpu_times()
            time.sleep(interval)
        else:
            t1 = _last_cpu_times_2
        _last_cpu_times_2 = cpu_times()
        return calculate(t1, _last_cpu_times_2)
    # per-cpu usage
    else:
        ret = []
        if blocking:
            tot1 = cpu_times(percpu=True)
            time.sleep(interval)
        else:
            tot1 = _last_per_cpu_times_2
        _last_per_cpu_times_2 = cpu_times(percpu=True)
        for t1, t2 in zip(tot1, _last_per_cpu_times_2):
            ret.append(calculate(t1, t2))
        return ret

# =====================================================================
# --- system memory related functions
# =====================================================================


def virtual_memory():
    """Return statistics about system memory usage as a namedtuple
    including the following fields, expressed in bytes:

     - total:
       total physical memory available.

     - available:
       the actual amount of available memory that can be given
       instantly to processes that request more memory in bytes; this
       is calculated by summing different memory values depending on
       the platform (e.g. free + buffers + cached on Linux) and it is
       supposed to be used to monitor actual memory usage in a cross
       platform fashion.

     - percent:
       the percentage usage calculated as (total - available) / total * 100

     - used:
       memory used, calculated differently depending on the platform and
       designed for informational purposes only:
        OSX: active + inactive + wired
        BSD: active + wired + cached
        LINUX: total - free

     - free:
       memory not being used at all (zeroed) that is readily available;
       note that this doesn't reflect the actual memory available
       (use 'available' instead)

    Platform-specific fields:

     - active (UNIX):
       memory currently in use or very recently used, and so it is in RAM.

     - inactive (UNIX):
       memory that is marked as not used.

     - buffers (BSD, Linux):
       cache for things like file system metadata.

     - cached (BSD, OSX):
       cache for various things.

     - wired (OSX, BSD):
       memory that is marked to always stay in RAM. It is never moved to disk.

     - shared (BSD):
       memory that may be simultaneously accessed by multiple processes.

    The sum of 'used' and 'available' does not necessarily equal total.
    On Windows 'available' and 'free' are the same.
    """
    return _psplatform.virtual_memory()


def swap_memory():
    """Return system swap memory statistics as a namedtuple including
    the following attributes:

     - total:   total swap memory in bytes
     - used:    used swap memory in bytes
     - free:    free swap memory in bytes
     - percent: the percentage usage
     - sin:     no. of bytes the system has swapped in from disk (cumulative)
     - sout:    no. of bytes the system has swapped out from disk (cumulative)

    'sin' and 'sout' on Windows are meaningless and always set to 0.
    """
    return _psplatform.swap_memory()

# =====================================================================
# --- disks/paritions related functions
# =====================================================================


def disk_usage(path):
    """Return disk usage statistics about the given path as a namedtuple
    including total, used and free space expressed in bytes plus the
    percentage usage.
    """
    return _psplatform.get_disk_usage(path)


def disk_partitions(all=False):
    """Return mounted partitions as a list of namedtuples including
    device, mount point, filesystem type and mount options (a raw
    string separated by commas which may vary depending on the platform).

    If "all" parameter is False return physical devices only and ignore
    all others.
    """
    return _psplatform.disk_partitions(all)


def disk_io_counters(perdisk=False):
    """Return system disk I/O statistics as a namedtuple including
    the following attributes:

     - read_count:  number of reads
     - write_count: number of writes
     - read_bytes:  number of bytes read
     - write_bytes: number of bytes written
     - read_time:   time spent reading from disk (in milliseconds)
     - write_time:  time spent writing to disk (in milliseconds)

    If perdisk is True return the same information for every
    physical disk installed on the system as a dictionary
    with partition names as the keys and the namedutuple
    described above as the values.

    On recent Windows versions 'diskperf -y' command may need to be
    executed first otherwise this function won't find any disk.
    """
    rawdict = _psplatform.disk_io_counters()
    if not rawdict:
        raise RuntimeError("couldn't find any physical disk")
    if perdisk:
        for disk, fields in rawdict.items():
            rawdict[disk] = _nt_disk_iostat(*fields)
        return rawdict
    else:
        return _nt_disk_iostat(*[sum(x) for x in zip(*rawdict.values())])

# =====================================================================
# --- network related functions
# =====================================================================


def net_io_counters(pernic=False):
    """Return network I/O statistics as a namedtuple including
    the following attributes:

     - bytes_sent:   number of bytes sent
     - bytes_recv:   number of bytes received
     - packets_sent: number of packets sent
     - packets_recv: number of packets received
     - errin:        total number of errors while receiving
     - errout:       total number of errors while sending
     - dropin:       total number of incoming packets which were dropped
     - dropout:      total number of outgoing packets which were dropped
                     (always 0 on OSX and BSD)

    If pernic is True return the same information for every
    network interface installed on the system as a dictionary
    with network interface names as the keys and the namedtuple
    described above as the values.
    """
    rawdict = _psplatform.net_io_counters()
    if not rawdict:
        raise RuntimeError("couldn't find any network interface")
    if pernic:
        for nic, fields in rawdict.items():
            rawdict[nic] = _nt_net_iostat(*fields)
        return rawdict
    else:
        return _nt_net_iostat(*[sum(x) for x in zip(*rawdict.values())])

# =====================================================================
# --- other system related functions
# =====================================================================


def get_boot_time():
    """Return the system boot time expressed in seconds since the epoch.
    This is also available as psutil.BOOT_TIME.
    """
    return _psplatform.get_system_boot_time()


def get_users():
    """Return users currently connected on the system as a list of
    namedtuples including the following attributes.

     - user: the name of the user
     - terminal: the tty or pseudo-tty associated with the user, if any.
     - host: the host name associated with the entry, if any.
     - started: the creation time as a floating point number expressed in
       seconds since the epoch.
    """
    return _psplatform.get_system_users()
