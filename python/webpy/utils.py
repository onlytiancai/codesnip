import re
import itertools



def group(seq, size):
    """
    Returns an iterator over a series of lists of length size from iterable.

        >>> list(group([1,2,3,4], 2))
        [[1, 2], [3, 4]]
        >>> list(group([1,2,3,4,5], 2))
        [[1, 2], [3, 4], [5]]
    """
    def take(seq, n):
        for i in xrange(n):
            yield seq.next()

    if not hasattr(seq, 'next'):
        seq = iter(seq)
    while True:
        x = list(take(seq, size))
        if x:
            yield x
        else:
            break


class _re_subm_proxy:
    def __init__(self):
        self.match = None

    def __call__(self, match):
        self.match = match
        return ''


re_compile = re.compile


def re_subm(pat, repl, string):
    """
    Like re.sub, but returns the replacement _and_ the match object.
    
        >>> t, m = re_subm('g(oo+)fball', r'f\\1lish', 'goooooofball')
        >>> t
        'foooooolish'
        >>> m.groups()
        ('oooooo',)
    """
    compiled_pat = re_compile(pat)
    proxy = _re_subm_proxy()
    compiled_pat.sub(proxy.__call__, string)
    return compiled_pat.sub(repl, string), proxy.match


def safestr(obj, encoding='utf-8'):
    r"""
    Converts any given object to utf-8 encoded string.
    
        >>> safestr('hello')
        'hello'
        >>> safestr(u'\u1234')
        '\xe1\x88\xb4'
        >>> safestr(2)
        '2'
    """
    if isinstance(obj, unicode):
        return obj.encode(encoding)
    elif isinstance(obj, str):
        return obj
    elif hasattr(obj, 'next'):  # iterator
        return itertools.imap(safestr, obj)
    else:
        return str(obj)
