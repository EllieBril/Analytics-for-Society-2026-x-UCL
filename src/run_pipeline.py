from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "logs" / ".mplconfig"))

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.build_analytic_datasets import build_2022_raw_datasets, build_multiyear_light, compute_country_gap_rebuild, write_equity_definition
    from src.correlates import analyze_correlates
    from src.country_analysis import analyze_country_trajectories
    from src.inventory import build_inventory
    from src.interventions import map_interventions
    from src.load_csv import load_country_trajectories, load_equity_gap, load_escs_trend, load_interventions
    from src.prototype import build_prototype
    from src.reporting import build_competition_outputs
    from src.risk_analysis import analyze_risk
    from src.segmentation import analyze_segmentation
    from src.utils import setup_logger
    from src.within_school import analyze_within_school
else:
    from .build_analytic_datasets import build_2022_raw_datasets, build_multiyear_light, compute_country_gap_rebuild, write_equity_definition
    from .correlates import analyze_correlates
    from .country_analysis import analyze_country_trajectories
    from .inventory import build_inventory
    from .interventions import map_interventions
    from .load_csv import load_country_trajectories, load_equity_gap, load_escs_trend, load_interventions
    from .prototype import build_prototype
    from .reporting import build_competition_outputs
    from .risk_analysis import analyze_risk
    from .segmentation import analyze_segmentation
    from .utils import setup_logger
    from .within_school import analyze_within_school


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SDG 4 equity analysis pipeline.")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--focus-year", type=int, default=2022)
    parser.add_argument("--min-school-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    logger = setup_logger(root / "logs" / "pipeline.log")
    logger.info("Starting SDG 4 pipeline at %s", root)

    inventory = build_inventory(root, root / "outputs" / "01_data_inventory", logger)

    multiyear = build_multiyear_light(root, root / "outputs" / "05_analytic_dataset", logger)
    supplied_gap = load_equity_gap(root)
    rebuilt_gap = compute_country_gap_rebuild(multiyear["students_dedup"])
    (root / "outputs" / "02_equity_definition").mkdir(parents=True, exist_ok=True)
    rebuilt_gap.to_csv(root / "outputs" / "02_equity_definition" / "computed_equity_gap_by_country_year.csv", index=False)
    write_equity_definition(root / "outputs" / "02_equity_definition", rebuilt_gap, supplied_gap, load_escs_trend(root))

    country = analyze_country_trajectories(
        rebuilt_gap=rebuilt_gap,
        supplied_gap=supplied_gap,
        supplied_trajectory=load_country_trajectories(root),
        output_dir=root / "outputs" / "03_country_trajectories",
        logger=logger,
    )

    analytic = build_2022_raw_datasets(root, root / "outputs" / "05_analytic_dataset", logger)
    risk = analyze_risk(analytic["school_profiles"], root / "outputs" / "04_school_risk", logger)
    correlates = analyze_correlates(risk["school_profiles"], root / "outputs" / "06_correlates", logger)
    within = analyze_within_school(risk["school_profiles"], root / "outputs" / "07_within_school_gap", logger)
    segmentation = analyze_segmentation(risk["school_profiles"], root / "outputs" / "08_segmentation", logger)
    interventions = map_interventions(segmentation["segment_assignments"], load_interventions(root), root / "outputs" / "09_intervention_mapping", logger)
    prototype = build_prototype(
        segmentation["segment_assignments"],
        interventions["mapping"],
        risk["school_profiles"],
        root / "outputs" / "10_prototype",
        logger,
    )
    build_competition_outputs(
        root=root,
        inventory=inventory["metadata_summary"],
        latest_rankings=country["latest_rankings"],
        trend_rankings=country["trend_rankings"],
        school_profiles=risk["school_profiles"],
        segment_profiles=segmentation["segment_profiles"],
        intervention_map=interventions["mapping"],
        final_dir=root / "outputs" / "12_final",
        competition_dir=root / "outputs" / "11_competition_fit",
        logger=logger,
    )
    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
