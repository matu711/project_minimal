#!/usr/bin/env python3
"""Resolve a Compose numeric identity without changing its UID, GID or HOME.

Probe as the service user; register missing entries as root inside ONLY the
selected container. No usermod/chown, image rebuild or container recreation.
"""
from __future__ import annotations

import argparse
import fcntl
import grp
import json
import os
from pathlib import Path
import pwd
import re

PASSWD = Path('/etc/passwd')
GROUP = Path('/etc/group')
LOCK = Path('/etc/.piper-user.lock')


def lookup(function, value):
    try:
        return function(value)
    except KeyError:
        return None


def probe():
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError('PiPER Bash/training must use a non-root service UID')
    if lookup(pwd.getpwuid, uid) is None or lookup(grp.getgrgid, gid) is None:
        print(json.dumps(dict(uid=uid, gid=gid, home=os.environ['HOME'])))


def available_name(preferred, numeric_id, function):
    base = preferred
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_.-]*', base):
        base = f'tuning_{numeric_id}'
    name = base
    suffix = 0
    while lookup(function, name) is not None:
        suffix += 1
        name = f'{base}_{numeric_id}' + (f'_{suffix}' if suffix > 1 else '')
    return name


def append_entry(path, entry):
    # Never truncate or rewrite an existing account database.
    with path.open('rb') as existing:
        content = existing.read()
    if content and not content.endswith(b'\n'):
        raise ValueError(f'{path} is missing its final newline; refusing to modify it')
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'ab') as output:
        output.write(entry.encode('utf-8'))
        output.flush()
        os.fsync(output.fileno())


def register(identity, username):
    if os.geteuid() != 0:
        raise PermissionError('Account registration requires root inside the tuning container')
    uid, gid, home = identity['uid'], identity['gid'], identity['home']
    if type(uid) is not int or type(gid) is not int or not 0 < uid < 2**32 - 1 or not 0 <= gid < 2**32 - 1:
        raise ValueError('Invalid non-root UID/GID')
    if not isinstance(home, str) or not home.startswith('/') or any(c in home for c in ':\n\r\x00'):
        raise ValueError('Invalid HOME')
    # Serialize concurrent run.sh launches and recheck NSS under the lock.
    descriptor = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if lookup(grp.getgrgid, gid) is None:
            group = available_name(f'tuning_{gid}', gid, grp.getgrnam)
            append_entry(GROUP, f'{group}:x:{gid}:\n')
        if lookup(pwd.getpwuid, uid) is None:
            name = available_name(username, uid, pwd.getpwnam)
            append_entry(PASSWD, f'{name}:x:{uid}:{gid}:PiPER runtime user:{home}:/bin/bash\n')
            if name != username:
                print(f'User name {username!r} unavailable; registered {name!r} for UID {uid}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--identity')
    parser.add_argument('--username')
    args = parser.parse_args()
    if args.probe:
        if args.identity is not None or args.username is not None:
            parser.error('--probe cannot be combined with registration arguments')
        probe()
    else:
        if args.identity is None or args.username is None:
            parser.error('--identity and --username are required for registration')
        register(json.loads(args.identity), args.username)


if __name__ == '__main__':
    main()
