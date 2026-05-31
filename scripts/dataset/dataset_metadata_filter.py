#!/usr/bin/env python3
"""
AstroPT Catalog Filter & Clean Dataset Generator
==================================================
Author: Senior Astro-Data Scientist & PyTorch/HPC Infrastructure Engineer
Description:
    Creates filtered FITS catalogs by applying two independent layers of filtering:
    
    Layer 1 - Metadata Expressions: Applies arbitrary algebraic quality filters on 
              FITS catalog columns (e.g., SNR thresholds, redshift cuts, flag masks).
              Uses the same expression syntax as AstroPT's `applied_filters` YAML config.
    
    Layer 2 - Image Quality Blacklist: Cross-references with the `dark_galaxies_registry.csv`
              from the flux analyzer audit to remove TargetIDs whose images are corrupted 
              (All-Black, Partially-Black, or NaN-Corrupted).
    
    The output is a new, self-contained FITS catalog that can be used directly as 
    `metadata_path` in the AstroPT training configuration, eliminating the need for 
    runtime `applied_filters` evaluation during dataloader initialization.

Usage:
    python catalog_filter.py \\
        --input_catalog /path/to/original.fits \\
        --output_catalog /path/to/filtered_clean.fits \\
        --filters '((Z < 0.15) && (SNR_SPEC_R > 3.0)) || ((Z >= 0.15) && (SNR_SPEC_Z > 3.0))' \\
        --dark_registry /path/to/dark_galaxies_registry.csv \\
        --remove_classes All-Black Partially-Black NaN-Corrupted
"""

import os
import sys
import re
import json
import logging
import argparse
import numpy as np
from astropy.table import Table

# Setup premium logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("astropt_catalog_filter")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="AstroPT Catalog Filter & Clean Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply SNR filter only:
  python catalog_filter.py --filters '((Z < 0.15) && (SNR_SPEC_R > 3.0)) || ((Z >= 0.15) && (SNR_SPEC_Z > 3.0))'

  # Apply SNR filter + remove dark images:
  python catalog_filter.py --filters '((Z < 0.15) && (SNR_SPEC_R > 3.0)) || ((Z >= 0.15) && (SNR_SPEC_Z > 3.0))' \\
      --dark_registry /path/to/dark_galaxies_registry.csv

  # Remove only partially-black images (no metadata filter):
  python catalog_filter.py --dark_registry /path/to/dark_galaxies_registry.csv --remove_classes Partially-Black
        """
    )
    parser.add_argument(
        "--input_catalog", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits",
        help="Path to the original input FITS catalog."
    )
    parser.add_argument(
        "--output_catalog", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits",
        help="Path to save the filtered output FITS catalog."
    )
    parser.add_argument(
        "--filters", type=str, nargs='+', default=[],
        help="One or more filter expressions to apply on catalog columns. "
             "Uses the same syntax as AstroPT applied_filters (e.g., '((Z < 0.15) && (SNR_SPEC_R > 3.0))')."
    )
    parser.add_argument(
        "--dark_registry", type=str, default=None,
        help="Path to the dark_galaxies_registry.csv from the flux analyzer audit. "
             "If provided, removes TargetIDs classified as corrupted image quality."
    )
    parser.add_argument(
        "--remove_classes", type=str, nargs='+',
        default=["All-Black", "Partially-Black", "NaN-Corrupted"],
        help="Quality class labels to remove when using --dark_registry. "
             "Default: All-Black Partially-Black NaN-Corrupted"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_metadata_filter",
        help="Directory to save the filtering summary report (JSON)."
    )
    return parser.parse_args()


def resolve_id_column(colnames: list) -> str:
    """Finds the target ID column name in the catalog."""
    for candidate in ['targetid', 'TARGETID', 'ID', 'id', 'OBJID', 'objid']:
        if candidate in colnames:
            return candidate
    raise KeyError(f"Could not find a target ID column in catalog columns: {colnames}")


def apply_metadata_filters(table: Table, filters: list) -> tuple:
    """
    Applies algebraic filter expressions to the FITS catalog.
    
    Uses numpy vectorized evaluation with restricted builtins for safety.
    Handles column name aliasing (Z <-> redshift) for compatibility with
    the AstroPT applied_filters YAML syntax.
    
    Args:
        table: Astropy Table with the catalog data.
        filters: List of filter expression strings.
        
    Returns:
        Tuple of (filtered_table, filter_report_dict)
    """
    initial_count = len(table)
    mask = np.ones(initial_count, dtype=bool)
    filter_reports = []
    
    for expr in filters:
        # Convert AstroPT filter syntax to Python/NumPy syntax
        py_expr = expr.replace('&&', '&').replace('||', '|')
        
        # Extract variable names from the expression
        found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
        
        # Build evaluation namespace from catalog columns only
        eval_namespace = {}
        for word in found_words:
            if word in table.colnames:
                eval_namespace[word] = np.array(table[word])
        
        # Handle Z <-> redshift aliasing
        if 'redshift' in table.colnames and 'Z' not in eval_namespace and 'Z' in found_words:
            eval_namespace['Z'] = np.array(table['redshift'])
        elif 'Z' in table.colnames and 'redshift' not in eval_namespace and 'redshift' in found_words:
            eval_namespace['redshift'] = np.array(table['Z'])
            
        # Evaluate filter with restricted builtins (no access to os, sys, etc.)
        try:
            expr_mask = eval(py_expr, {"np": np, "__builtins__": {}}, eval_namespace)
            expr_mask = np.asarray(expr_mask, dtype=bool)
        except Exception as e:
            logger.error(f"Failed to evaluate filter expression: '{expr}' -> Error: {e}")
            raise ValueError(f"Invalid filter expression: '{expr}'. Error: {e}")
        
        passed = int(np.sum(expr_mask))
        rejected = initial_count - passed
        
        filter_reports.append({
            "expression": expr,
            "passed": passed,
            "rejected": rejected,
            "retention_rate": passed / initial_count if initial_count > 0 else 0.0
        })
        
        logger.info(f"  Filter: '{expr}'")
        logger.info(f"    Passed: {passed:,} | Rejected: {rejected:,} | Retention: {passed/initial_count:.1%}")
        
        mask &= expr_mask
    
    # Apply combined mask
    final_count = int(np.sum(mask))
    table_filtered = table[mask]
    
    report = {
        "initial_count": initial_count,
        "final_count": final_count,
        "total_removed": initial_count - final_count,
        "combined_retention_rate": final_count / initial_count if initial_count > 0 else 0.0,
        "individual_filters": filter_reports
    }
    
    return table_filtered, report


def apply_image_quality_filter(table: Table, dark_registry_path: str, 
                                remove_classes: list, id_col: str) -> tuple:
    """
    Cross-references the FITS catalog with the flux analyzer's dark_galaxies_registry.csv
    and removes TargetIDs classified under the specified quality classes.
    
    Args:
        table: Astropy Table (potentially already metadata-filtered).
        dark_registry_path: Path to the dark_galaxies_registry.csv.
        remove_classes: List of quality class labels to remove.
        id_col: Name of the ID column in the catalog.
        
    Returns:
        Tuple of (filtered_table, image_filter_report_dict)
    """
    import pandas as pd
    
    initial_count = len(table)
    
    # Load the dark registry
    logger.info(f"  Loading image quality audit registry: {dark_registry_path}")
    df_dark = pd.read_csv(dark_registry_path)
    
    total_anomalies_in_registry = len(df_dark)
    logger.info(f"  Total anomalous entries in registry: {total_anomalies_in_registry:,}")
    
    # Filter to the requested quality classes
    class_mask = df_dark['quality_class'].isin(remove_classes)
    df_to_remove = df_dark[class_mask]
    
    # Log per-class breakdown
    class_breakdown = {}
    for cls in remove_classes:
        count = int((df_dark['quality_class'] == cls).sum())
        class_breakdown[cls] = count
        logger.info(f"    {cls}: {count:,} entries in registry")
    
    # Extract the set of bad TargetIDs
    bad_ids = set(df_to_remove['TargetID'].astype(np.int64).values)
    logger.info(f"  Total unique TargetIDs to blacklist: {len(bad_ids):,}")
    
    # Build the keep mask
    catalog_ids = np.array(table[id_col], dtype=np.int64)
    keep_mask = ~np.isin(catalog_ids, list(bad_ids))
    
    matched_removals = int(np.sum(~keep_mask))
    table_filtered = table[keep_mask]
    final_count = len(table_filtered)
    
    logger.info(f"  Matched and removed from catalog: {matched_removals:,}")
    logger.info(f"  Catalog after image filter: {final_count:,} records")
    
    # Coverage warning
    if total_anomalies_in_registry < initial_count * 0.01:
        logger.warning(
            f"  ⚠ LOW AUDIT COVERAGE: The dark registry contains only {total_anomalies_in_registry:,} "
            f"anomalies from what appears to be a partial audit. For comprehensive filtering, "
            f"run the full flux audit with --max_samples -1 first."
        )
    
    report = {
        "registry_path": dark_registry_path,
        "total_anomalies_in_registry": total_anomalies_in_registry,
        "quality_classes_removed": remove_classes,
        "class_breakdown": class_breakdown,
        "unique_bad_ids": len(bad_ids),
        "matched_removals_from_catalog": matched_removals,
        "unmatched_bad_ids": len(bad_ids) - matched_removals,
        "initial_count": initial_count,
        "final_count": final_count,
        "retention_rate": final_count / initial_count if initial_count > 0 else 0.0
    }
    
    return table_filtered, report


def main():
    args = parse_args()
    
    # -------------------------------------------------------------------------
    # STAGE 0: Setup
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure file logging
    log_file = os.path.join(args.output_dir, "catalog_filter.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("ASTROPT CATALOG FILTER & CLEAN DATASET GENERATOR")
    logger.info("=" * 80)
    logger.info(f"Input catalog:  {args.input_catalog}")
    logger.info(f"Output catalog: {args.output_catalog}")
    logger.info(f"Metadata filters: {args.filters if args.filters else '(none)'}")
    logger.info(f"Dark registry:  {args.dark_registry if args.dark_registry else '(none)'}")
    logger.info(f"Remove classes: {args.remove_classes}")
    
    # -------------------------------------------------------------------------
    # STAGE 1: Load Original FITS Catalog
    # -------------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Stage 1: Loading original FITS catalog...")
    
    if not os.path.exists(args.input_catalog):
        logger.error(f"Input catalog not found: {args.input_catalog}")
        sys.exit(1)
    
    t = Table.read(args.input_catalog)
    original_count = len(t)
    id_col = resolve_id_column(t.colnames)
    
    logger.info(f"  Loaded {original_count:,} records with {len(t.colnames)} columns.")
    logger.info(f"  ID column resolved as: '{id_col}'")
    
    # Summary report accumulator
    summary = {
        "input_catalog": args.input_catalog,
        "output_catalog": args.output_catalog,
        "original_record_count": original_count,
        "id_column": id_col,
        "metadata_filter_report": None,
        "image_quality_filter_report": None,
        "final_record_count": None,
        "overall_retention_rate": None
    }
    
    # -------------------------------------------------------------------------
    # STAGE 2: Apply Metadata-Level Expression Filters
    # -------------------------------------------------------------------------
    if args.filters:
        logger.info("-" * 50)
        logger.info(f"Stage 2: Applying {len(args.filters)} metadata filter expression(s)...")
        
        t, meta_report = apply_metadata_filters(t, args.filters)
        summary["metadata_filter_report"] = meta_report
        
        logger.info(f"  After metadata filters: {len(t):,} / {original_count:,} records "
                     f"({len(t)/original_count:.1%} retained)")
    else:
        logger.info("-" * 50)
        logger.info("Stage 2: No metadata filters specified. Skipping.")
    
    # -------------------------------------------------------------------------
    # STAGE 3: Apply Image Quality Blacklist (Dark Registry)
    # -------------------------------------------------------------------------
    if args.dark_registry:
        logger.info("-" * 50)
        logger.info("Stage 3: Applying image quality blacklist from flux audit registry...")
        
        if not os.path.exists(args.dark_registry):
            logger.error(f"Dark registry CSV not found: {args.dark_registry}")
            sys.exit(1)
        
        t, img_report = apply_image_quality_filter(t, args.dark_registry, args.remove_classes, id_col)
        summary["image_quality_filter_report"] = img_report
    else:
        logger.info("-" * 50)
        logger.info("Stage 3: No dark registry provided. Skipping image quality filter.")
    
    # -------------------------------------------------------------------------
    # STAGE 4: Save Filtered FITS Catalog
    # -------------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Stage 4: Saving filtered FITS catalog...")
    
    final_count = len(t)
    summary["final_record_count"] = final_count
    summary["overall_retention_rate"] = final_count / original_count if original_count > 0 else 0.0
    
    # Ensure output directory exists
    output_catalog_dir = os.path.dirname(args.output_catalog)
    if output_catalog_dir:
        os.makedirs(output_catalog_dir, exist_ok=True)
    
    t.write(args.output_catalog, overwrite=True)
    
    logger.info(f"  Filtered catalog saved to: {args.output_catalog}")
    logger.info(f"  Final record count: {final_count:,} / {original_count:,} "
                f"({final_count/original_count:.1%} retained)")
    
    # -------------------------------------------------------------------------
    # STAGE 5: Generate Summary Report
    # -------------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Stage 5: Generating filtering summary report...")
    
    summary_path = os.path.join(args.output_dir, "catalog_filter_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    logger.info(f"  Summary report saved to: {summary_path}")
    
    # -------------------------------------------------------------------------
    # Final Summary Log
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("CATALOG FILTERING COMPLETE")
    logger.info(f"  Original:  {original_count:,} records")
    
    if args.filters:
        after_meta = summary["metadata_filter_report"]["final_count"]
        logger.info(f"  After SNR/metadata filters: {after_meta:,} records "
                     f"(-{original_count - after_meta:,})")
    
    if args.dark_registry and summary["image_quality_filter_report"]:
        img_removed = summary["image_quality_filter_report"]["matched_removals_from_catalog"]
        logger.info(f"  After image quality filter: {final_count:,} records (-{img_removed:,})")
    
    logger.info(f"  FINAL CLEAN CATALOG: {final_count:,} records ({final_count/original_count:.1%} retained)")
    logger.info(f"  Output: {args.output_catalog}")
    logger.info("=" * 80)
    
    # Print the recommended training config update
    logger.info("")
    logger.info("RECOMMENDED AstroPT YAML CONFIG UPDATE:")
    logger.info("  metadata_path: \"%s\"" % args.output_catalog)
    logger.info("  applied_filters: []  # Already baked into the filtered catalog")
    logger.info("")


if __name__ == "__main__":
    main()
