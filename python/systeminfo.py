import psutil

def get_sysinfo():
    result = {}
    result['cpu_utilization'] = psutil.cpu_percent()
    result['mem_utilization'] = psutil.virtual_memory().percent
    result['swap_utilization'] = psutil.swap_memory().percent
    for part in psutil.disk_partitions(all=False):
        r = psutil.disk_usage(part.mountpoint)
        result['disk_utilization(%s)' % part.mountpoint] = r.percent
    for k, v in psutil.disk_io_counters(True).items():
        if k.startswith('dm'):
            continue
        result['disk_io_read_bytes(%s)' % k] = v.read_bytes
        result['disk_io_write_bytes(%s)' % k] = v.write_bytes
    for k, v in psutil.net_io_counters(True).items():
        if not k.startswith('eth'):
            continue
        result['net_io_bytes_sent(%s)' % k] = v.bytes_sent
        result['net_io_bytes_recv(%s)' % k] = v.bytes_recv
    return result



if __name__ == '__main__':
    import pprint
    pprint.pprint(get_sysinfo())
