def pretty_bytes(self, size):
    ranges = ((1 << 70L, 'ZB'),
              (1 << 60L, 'EB'),
              (1 << 50L, 'PB'),
              (1 << 40L, 'TB'),
              (1 << 30L, 'GB'),
              (1 << 20L, 'MB'),
              (1 << 10L, 'KB'),
              (1, 'Bytes'))
    for limit, suffix in ranges:
        if size >= limit:
            break
    return '%.2f %s' % (float(size) / limit, suffix)
