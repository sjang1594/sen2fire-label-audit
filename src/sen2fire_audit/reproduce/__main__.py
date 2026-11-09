"""Usage: python -m sen2fire_audit.reproduce <name> [--data DIR] [--results DIR] [--out DIR] [args...]"""
import argparse
import runpy
import sys

from . import cmd, _paths

def main():
    ap = argparse.ArgumentParser(prog='python -m sen2fire_audit.reproduce', description=__doc__,
                                 epilog='commands: ' + ', '.join(cmd))
    ap.add_argument('name', choices=sorted(cmd))
    ap.add_argument('--data', help='dataset root (default: ./data or $SEN2FIRE_AUDIT_DATA)')
    ap.add_argument('--results', help='released result tables (default: ./results or $SEN2FIRE_AUDIT_RESULTS)')
    ap.add_argument('--out', help='output directory (default: <results>/analysis)')
    args, rest = ap.parse_known_args()
    _paths.configure(data=args.data, results=args.results, out=args.out)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.argv = [args.name] + rest
    runpy.run_module(f'{__package__}.{args.name}', run_name='__main__', alter_sys=True)

if __name__ == '__main__':
    main()
