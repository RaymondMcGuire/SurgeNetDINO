"""Compatibility entry point; importing this file never creates/downloads models."""

if __name__ == "__main__":
    import sys

    from surgenet_dino.cli import main

    args = sys.argv[1:]
    commands = {"list", "plan", "doctor", "smoke", "encode"}
    if not args or (args[0] not in commands and args[0] not in {"-h", "--help"}):
        args = ["smoke", *args]
    raise SystemExit(main(args))
