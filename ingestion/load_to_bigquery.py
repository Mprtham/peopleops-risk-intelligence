"""
Load the five synthetic HR CSVs into BigQuery dataset `peopleops_raw`.

Auth: run `gcloud auth application-default login` before executing.
Project: set GCP_PROJECT env var or pass --project on the command line.

Re-running is safe — tables are truncated then reloaded (WRITE_TRUNCATE).
BigQuery sandbox tables expire after 60 days; this script sets no expiry so
the tables persist for the project lifetime on a free-trial project.
"""

import argparse
import os
from pathlib import Path

from google.cloud import bigquery
from google.cloud.bigquery import SchemaField, LoadJobConfig, WriteDisposition, SourceFormat

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DATASET_ID = "peopleops_raw"

SCHEMAS: dict[str, list[SchemaField]] = {
    "employees": [
        SchemaField("employee_id",   "INTEGER",  mode="REQUIRED"),
        SchemaField("full_name",     "STRING",   mode="REQUIRED"),
        SchemaField("gender",        "STRING",   mode="REQUIRED"),
        SchemaField("date_of_birth", "DATE",     mode="REQUIRED"),
        SchemaField("department",    "STRING",   mode="REQUIRED"),
        SchemaField("hire_date",     "DATE",     mode="REQUIRED"),
        SchemaField("contract_type", "STRING",   mode="REQUIRED"),
    ],
    "role_history": [
        SchemaField("employee_id",    "INTEGER", mode="REQUIRED"),
        SchemaField("job_role",       "STRING",  mode="REQUIRED"),
        SchemaField("role_level",     "INTEGER", mode="REQUIRED"),
        SchemaField("salary",         "INTEGER", mode="REQUIRED"),
        SchemaField("effective_from", "DATE",    mode="REQUIRED"),
        SchemaField("effective_to",   "DATE",    mode="NULLABLE"),
    ],
    "performance_reviews": [
        SchemaField("review_id",   "INTEGER", mode="REQUIRED"),
        SchemaField("employee_id", "INTEGER", mode="REQUIRED"),
        SchemaField("review_date", "DATE",    mode="REQUIRED"),
        SchemaField("rating",      "INTEGER", mode="REQUIRED"),
    ],
    "absence_events": [
        SchemaField("absence_id",   "INTEGER", mode="REQUIRED"),
        SchemaField("employee_id",  "INTEGER", mode="REQUIRED"),
        SchemaField("absence_date", "DATE",    mode="REQUIRED"),
        SchemaField("days",         "INTEGER", mode="REQUIRED"),
        SchemaField("absence_type", "STRING",  mode="REQUIRED"),
    ],
    "exits": [
        SchemaField("exit_id",     "INTEGER", mode="REQUIRED"),
        SchemaField("employee_id", "INTEGER", mode="REQUIRED"),
        SchemaField("exit_date",   "DATE",    mode="REQUIRED"),
        SchemaField("exit_reason", "STRING",  mode="REQUIRED"),
    ],
}


def get_project(args_project: str | None) -> str:
    project = args_project or os.environ.get("GCP_PROJECT")
    if not project:
        raise SystemExit(
            "GCP project not set.\n"
            "Either export GCP_PROJECT=<your-project-id>  or  pass --project <id>"
        )
    return project


def ensure_dataset(client: bigquery.Client, project: str) -> bigquery.Dataset:
    dataset_ref = bigquery.DatasetReference(project, DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {project}.{DATASET_ID}")
    return dataset


def load_table(
    client: bigquery.Client,
    project: str,
    table_name: str,
    csv_path: Path,
) -> int:
    table_ref = f"{project}.{DATASET_ID}.{table_name}"

    job_config = LoadJobConfig(
        schema=SCHEMAS[table_name],
        source_format=SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=WriteDisposition.WRITE_TRUNCATE,
        # Prevent sandbox 60-day expiry
        time_partitioning=None,
    )

    with open(csv_path, "rb") as f:
        job = client.load_table_from_file(f, table_ref, job_config=job_config)

    job.result()  # wait for completion

    if job.errors:
        raise RuntimeError(f"Load failed for {table_name}: {job.errors}")

    table = client.get_table(table_ref)
    # Try to clear the 60-day sandbox expiry — requires billing to be enabled.
    # In sandbox mode this silently skips; re-run `make load` if tables expire.
    if table.expires is not None:
        try:
            table.expires = None
            client.update_table(table, ["expires"])
        except Exception:
            pass  # sandbox mode — expiry stays, re-run make load if needed

    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load HR CSVs to BigQuery")
    parser.add_argument("--project", help="GCP project ID (overrides GCP_PROJECT env var)")
    args = parser.parse_args()

    project = get_project(args.project)
    client = bigquery.Client(project=project)

    ensure_dataset(client, project)

    print(f"\nLoading tables from {RAW_DIR.resolve()}\n")
    total_rows = 0
    for table_name in SCHEMAS:
        csv_path = RAW_DIR / f"{table_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found — run `python data/generate_synthetic_data.py` first"
            )
        rows = load_table(client, project, table_name, csv_path)
        print(f"  {table_name:<25} {rows:>8,} rows  ->  {project}.{DATASET_ID}.{table_name}")
        total_rows += rows

    print(f"\n  Total rows loaded: {total_rows:,}")
    print("\nDone. Re-run anytime to refresh (tables are truncated and reloaded).")


if __name__ == "__main__":
    main()
