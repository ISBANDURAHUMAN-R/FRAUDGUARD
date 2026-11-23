"""Tests for SQLite Database Storage, Search, Filtering, and Pagination.

Author: Isbandu Rahuman R.
"""

from pathlib import Path
import pytest

from src.database import (
    clear_history,
    delete_prediction,
    generate_transaction_id,
    get_history,
    get_history_stats,
    init_db,
    insert_bulk_predictions,
    insert_prediction,
)


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated temporary SQLite database for tests."""
    db_file = tmp_path / "test_fraudguard.db"
    init_db(db_path=db_file)
    return db_file


def test_insert_and_get_single_prediction(test_db):
    """Test inserting a single record and retrieving it."""
    tx_id = generate_transaction_id("TX-UNIT-")
    inserted_id = insert_prediction(
        transaction_id=tx_id,
        prediction_source="single",
        amount=149.50,
        time_sec=500.0,
        prediction_label="Legitimate",
        fraud_probability=0.042,
        risk_level="LOW",
        notes="Automated test note",
        db_path=test_db
    )

    assert inserted_id == tx_id

    records, total = get_history(search=tx_id, db_path=test_db)
    assert total == 1
    assert len(records) == 1
    row = records[0]
    assert row["transaction_id"] == tx_id
    assert row["amount"] == 149.50
    assert row["prediction_label"] == "Legitimate"
    assert row["risk_level"] == "LOW"


def test_insert_bulk_and_stats(test_db):
    """Test bulk insertion and aggregate summary metrics."""
    bulk_records = [
        {
            "transaction_id": generate_transaction_id("TX-BLK-"),
            "prediction_source": "bulk",
            "amount": 100.0,
            "time_sec": 10.0,
            "prediction_label": "Legitimate",
            "fraud_probability": 0.05,
            "risk_level": "LOW"
        },
        {
            "transaction_id": generate_transaction_id("TX-BLK-"),
            "prediction_source": "bulk",
            "amount": 850.0,
            "time_sec": 20.0,
            "prediction_label": "Fraudulent",
            "fraud_probability": 0.88,
            "risk_level": "CRITICAL"
        }
    ]

    count = insert_bulk_predictions(bulk_records, db_path=test_db)
    assert count == 2

    stats = get_history_stats(db_path=test_db)
    assert stats["total_predictions"] == 2
    assert stats["fraud_count"] == 1
    assert stats["legit_count"] == 1
    assert stats["fraud_rate"] == 50.0
    assert stats["total_volume_analyzed"] == 950.0


def test_filters_and_pagination(test_db):
    """Test risk level filter and pagination."""
    # Insert 5 low risk and 3 high risk
    for i in range(5):
        insert_prediction(
            transaction_id=f"TX-LOW-{i}",
            prediction_source="single",
            amount=50.0 + i,
            time_sec=float(i),
            prediction_label="Legitimate",
            fraud_probability=0.05,
            risk_level="LOW",
            db_path=test_db
        )
    for j in range(3):
        insert_prediction(
            transaction_id=f"TX-HIGH-{j}",
            prediction_source="bulk",
            amount=500.0 + j,
            time_sec=float(j),
            prediction_label="Fraudulent",
            fraud_probability=0.75,
            risk_level="HIGH",
            db_path=test_db
        )

    # Filter LOW
    low_records, low_total = get_history(risk_filter="LOW", db_path=test_db)
    assert low_total == 5
    assert all(r["risk_level"] == "LOW" for r in low_records)

    # Filter HIGH
    high_records, high_total = get_history(risk_filter="HIGH", db_path=test_db)
    assert high_total == 3

    # Pagination: limit 2, offset 0
    p1_records, total = get_history(limit=2, offset=0, db_path=test_db)
    assert total == 8
    assert len(p1_records) == 2


def test_delete_and_clear_history(test_db):
    """Test deleting individual records and purging database."""
    tx_id = generate_transaction_id("TX-DEL-")
    insert_prediction(
        transaction_id=tx_id,
        prediction_source="single",
        amount=25.0,
        time_sec=1.0,
        prediction_label="Legitimate",
        fraud_probability=0.01,
        risk_level="LOW",
        db_path=test_db
    )

    deleted = delete_prediction(tx_id, db_path=test_db)
    assert deleted is True

    records, total = get_history(search=tx_id, db_path=test_db)
    assert total == 0

    # Test clear_history
    insert_prediction(
        transaction_id="TX-KEEP-1",
        prediction_source="single",
        amount=10.0,
        time_sec=1.0,
        prediction_label="Legitimate",
        fraud_probability=0.02,
        risk_level="LOW",
        db_path=test_db
    )
    cleared_count = clear_history(db_path=test_db)
    assert cleared_count == 1
    _, after_clear_total = get_history(db_path=test_db)
    assert after_clear_total == 0
