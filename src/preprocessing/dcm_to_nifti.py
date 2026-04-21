#!/usr/bin/env python3
"""
DICOM → NIfTI Conversion (CSV-driven, parallel)
=================================================
Reads a cohort CSV, converts each DICOM series at the specified
path column into NIfTI using parallel workers, and saves an
updated CSV with the new paths.

Usage:
    python convert_dcm_to_nifti.py \
        --cohort-csv cohort.csv \
        --path-col mri_path \
        --output-dir /data/nifti \
        --workers 8 \
        --skip-existing
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def convert_dicom_series_to_nifti(dicom_dir: str, output_file: str) -> bool:
    """
    Convert a single DICOM series to NIfTI.
    Uses GetGDCMSeriesFileNames (reads DICOM headers) instead of
    globbing for *.dcm, since ADNI files don't always have .dcm extension.
    """
    try:
        reader = sitk.ImageSeriesReader()

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
        if not series_ids:
            return False

        if len(series_ids) > 1:
            print(f"  WARNING: {len(series_ids)} series in {dicom_dir}, using first")

        dicom_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dicom_dir, series_ids[0]
        )
        if not dicom_files:
            return False

        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        sitk.WriteImage(image, output_file)
        return True

    except Exception as e:
        print(f"  Error converting {dicom_dir}: {e}")
        return False


def convert_one(args_tuple):
    """Wrapper for parallel execution. Takes (idx, dicom_dir, output_path)."""
    idx, dicom_dir, output_path = args_tuple
    success = convert_dicom_series_to_nifti(dicom_dir, output_path)
    return idx, output_path, success


def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to NIfTI, driven by a cohort CSV."
    )
    parser.add_argument("--cohort-csv", required=True, help="Input cohort CSV")
    parser.add_argument("--path-col", required=True, help="Column with DICOM directory paths")
    parser.add_argument("--output-dir", required=True, help="Directory for NIfTI output files")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: adds _nifti suffix)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output NIfTI already exists")
    args = parser.parse_args()

    if args.output_csv is None:
        base = os.path.splitext(args.cohort_csv)[0]
        args.output_csv = f"{base}_nifti.csv"

    # Load
    df = pd.read_csv(args.cohort_csv)
    print(f"Loaded {len(df):,} rows from {args.cohort_csv}")

    if args.path_col not in df.columns:
        print(f"ERROR: Column '{args.path_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build jobs
    df["nifti_path"] = ""
    jobs = []

    for idx, row in df.iterrows():
        dicom_dir = str(row[args.path_col])

        if dicom_dir in ("NOT_FOUND", "nan", ""):
            df.at[idx, "nifti_path"] = "NOT_FOUND"
            continue

        # Filename from subject_id + image_id if available, else row index
        if "subject_id" in df.columns and "image_id_mri" in df.columns:
            sid = str(row["subject_id"]).replace("/", "_")
            iid = str(int(row["image_id_mri"])) if pd.notna(row["image_id_mri"]) else str(idx)
            output_name = f"{sid}_{iid}.nii.gz"
        else:
            output_name = f"{idx}.nii.gz"

        output_path = os.path.join(args.output_dir, output_name)

        if args.skip_existing and os.path.exists(output_path):
            df.at[idx, "nifti_path"] = output_path
            continue

        if not os.path.isdir(dicom_dir):
            df.at[idx, "nifti_path"] = "DIR_NOT_FOUND"
            continue

        jobs.append((idx, dicom_dir, output_path))

    n_skipped = len(df) - len(jobs)
    print(f"Jobs to run: {len(jobs):,}  (skipped: {n_skipped:,})")

    if not jobs:
        df.to_csv(args.output_csv, index=False)
        print("Nothing to convert. CSV saved.")
        return

    # Run parallel conversion
    successful, failed = 0, 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, job): job for job in jobs}

        for future in tqdm(as_completed(futures), total=len(jobs), desc="Converting"):
            idx, output_path, success = future.result()
            if success:
                df.at[idx, "nifti_path"] = output_path
                successful += 1
            else:
                df.at[idx, "nifti_path"] = "CONVERSION_FAILED"
                failed += 1

    elapsed = time.time() - start

    # Save
    df.to_csv(args.output_csv, index=False)
    print(f"\nDone in {elapsed / 60:.1f} min — {successful} converted, {failed} failed, {n_skipped} skipped")
    print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()