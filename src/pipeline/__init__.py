"""
AWS Lambda pipeline handlers.

    daily_pipeline   — Daily batch: retrain models + backfill predictions
    hourly_pipeline  — Hourly inference: ingest → predict → store to DynamoDB
"""