"""Command-line entrypoint for ICC, SEM, and MDC analyses.

Subcommands process repeated wide-format ratings and reader-study ratings.
Outputs are written to new files without replacing existing results.
"""

import argparse
from pathlib import Path

import pandas as pd

from reliability.wide_repeatability import run_wide_repeatability
from reliability.intra_rater import intra_rater_repeatability_table
from reliability.reader_icc import inter_rater_icc_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='analysis', required=True)
    wide = commands.add_parser('wide-repeatability', help='ICC and MDC from paired wide-format ratings')
    wide.add_argument('--input-dir', type=Path, required=True)
    wide.add_argument('--output-dir', type=Path, required=True)
    wide.add_argument('--seed', type=int, default=None)
    reader = commands.add_parser('inter-rater', help='Reader ICC from canonical long ratings')
    reader.add_argument('--input-csv', type=Path, required=True)
    reader.add_argument('--output-csv', type=Path, required=True)
    reader.add_argument('--n-bootstrap', type=int, default=2000)
    reader.add_argument('--seed', type=int, default=42)
    reader.add_argument('--reader-scope', choices=['random', 'fixed'], default='random',
                        help='Generalize to comparable readers, or restrict inference to these readers')
    repeat = commands.add_parser('intra-rater', help='Same-reader, same-condition repeated-session ICC and MDC')
    repeat.add_argument('--input-csv', type=Path, required=True)
    repeat.add_argument('--output-csv', type=Path, required=True)
    repeat.add_argument('--n-bootstrap', type=int, default=2000)
    repeat.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.analysis == 'wide-repeatability':
        run_wide_repeatability(args.input_dir, args.output_dir, args.seed)
        return
    if args.output_csv.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_csv}')
    data = pd.read_csv(args.input_csv, dtype={
        'case_id': str, 'patient_id': str, 'reader_id': str, 'session_id': str,
        'task': str, 'outcome': str, 'method': str,
    })
    result = (inter_rater_icc_table(data, args.n_bootstrap, args.seed, args.reader_scope)
              if args.analysis == 'inter-rater'
              else intra_rater_repeatability_table(data, args.n_bootstrap, args.seed))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('x', encoding='utf-8', newline='') as handle:
        result.to_csv(handle, index=False)


if __name__ == '__main__':
    main()
