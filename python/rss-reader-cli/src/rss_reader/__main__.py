import argparse
from .app import RSSReaderApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSS Reader CLI")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (can use -v, -vv)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors")
    args = parser.parse_args()

    import logging
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.getLogger().setLevel(level)

    app = RSSReaderApp()
    app.run()
