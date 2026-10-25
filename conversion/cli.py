"""Convert single/dual PiPER MCAP or synchronized raw recordings to LeRobot v2.1."""
import argparse
import json
from pathlib import Path
import sys
from conversion.pipeline import convert

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None, default_robot=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['single','bimanual'], default=default_robot, required=default_robot is None)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--input', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--task', help='Recorded task override; required when input contains no task metadata')
    parser.add_argument('--mcap-schema', type=Path)
    parser.add_argument('--validate-only', action='store_true', help='Check input without writing a dataset')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--timing-log', type=Path)
    args, extra = parser.parse_known_args(argv)
    if extra and args.robot != "single":
        parser.error("Unsupported bimanual arguments: " + " ".join(extra))
    if Path(args.dataset).name != args.dataset or args.dataset in ('.','..'):
        parser.error('--dataset must be a single directory name')
    try:
        result = convert(args.robot, args.input or [ROOT/'datasets/mcap'/args.dataset],
                         args.output or ROOT/'datasets/lerobot/local'/args.dataset,
                         f'local/{args.dataset}', args.task, args.mcap_schema, args.validate_only, single_options=extra, resize=args.resize, timing_log=args.timing_log)
    except (ValueError, OSError, RuntimeError, KeyError, AssertionError) as error:
        print(f'Conversion failed: {error}', file=sys.stderr)
        for note in getattr(error, '__notes__', []):
            print(note, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
