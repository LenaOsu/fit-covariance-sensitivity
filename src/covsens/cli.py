"""Command line:  covsens generate | covsens run --config configs/demo.yaml"""
from __future__ import annotations

import argparse
import logging

from covsens.pipeline import run
from covsens.synthetic import make_synthetic_dataset, save_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="covsens", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the synthetic benchmark dataset")
    g.add_argument("--out", default="data/synthetic/synthetic_fit.npz")
    g.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run", help="run the full analysis from a YAML config")
    r.add_argument("--config", default="configs/demo.yaml")
    s = sub.add_parser("surrogate", help="ML extension: benchmark surrogates against exact toys")
    s.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "generate":
        ds, resp = make_synthetic_dataset(seed=args.seed)
        save_dataset(args.out, ds, resp)
        logging.getLogger("covsens").info("wrote %s (%d parameters, %d bins)", args.out, ds.fit.n_params, ds.n_bins)
    elif args.cmd == "surrogate":
        from covsens.extensions.surrogate import run as run_surrogate  # lazy: needs scikit-learn

        table = run_surrogate(args.config)
        cols = ["model", "n_train", "scenario", "sigma_err_median_pct", "ratio_err_median_pct", "ratio_err_max_pct"]
        print(table[cols].round(2).to_string(index=False))
    else:
        res = run(args.config)
        print("\n" + res["summary"].round(3).to_string(index=False))
        print("\nTop-10 most sensitive bins (headline scenario):")
        cols = ["sample", "variable", "bin_lo", "bin_hi", "rel_ref_pct", "rel_mod_pct", "sigma_ratio"]
        print(res["top"][cols].round(3).to_string(index=False))
