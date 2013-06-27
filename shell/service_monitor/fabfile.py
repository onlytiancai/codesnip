from __future__ import with_statement
from fabric.api import local, settings, abort, run
from fabric.contrib.console import confirm

def ensure_deploy_dir_exist():
    with settings(warn_only=True):
        result = run('test -d /data/applications', capture=True)
    if result.failed and not confirm("Tests failed. Continue anyway?"):
        abort("Aborting at user request.")
