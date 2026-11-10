"""Peer credential checks for the Unix socket.

Filesystem permissions (0700 directory, 0600 socket) are the primary control;
this is defence in depth against a process running as another user.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import socket
import struct
import sys


def peer_uid(sock: socket.socket) -> int | None:
    """Return the uid of the process on the other end of the socket.

    Returns None when the platform does not offer a way to find it out.
    """
    if sys.platform.startswith("linux"):
        try:
            raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        except OSError:
            return None
        pid, uid, _gid = struct.unpack("3i", raw)
        return uid

    # macOS / BSD: getpeereid is not exposed by the socket module, but libc
    # has it and avoids depending on the xucred struct layout.
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return None
    try:
        libc = ctypes.CDLL(libc_path, use_errno=True)
        libc.getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        libc.getpeereid.restype = ctypes.c_int
        uid = ctypes.c_uint(0)
        gid = ctypes.c_uint(0)
        if libc.getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
            return None
        return int(uid.value)
    except (OSError, AttributeError):
        return None
