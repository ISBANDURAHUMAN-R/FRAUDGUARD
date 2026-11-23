"""SQLite Transaction History Database Module for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

DEFAULT_DB_PATH = Path("data") / "fraudguard.db"


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Create a thread-safe connection to SQLite database."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize database tables and indexes."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            prediction_source TEXT NOT NULL,
            amount REAL NOT NULL,
            time_sec REAL NOT NULL,
            prediction_label TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            notes TEXT
        );
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_timestamp 
        ON transactions(timestamp DESC);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_risk 
        ON transactions(risk_level);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_source 
        ON transactions(prediction_source);
    """)

    conn.commit()
    conn.close()


def generate_transaction_id(prefix: str = "TX-") -> str:
    """Generate a clean unique transaction reference."""
    return f"{prefix}{uuid.uuid4().hex[:10].upper()}"


def insert_prediction(
    transaction_id: str,
    prediction_source: str,
    amount: float,
    time_sec: float,
    prediction_label: str,
    fraud_probability: float,
    risk_level: str,
    timestamp: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> str:
    """Insert a single prediction record using safe parameterized SQL."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    ts = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO transactions (
            transaction_id, timestamp, prediction_source, amount,
            time_sec, prediction_label, fraud_probability, risk_level, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(transaction_id),
        ts,
        str(prediction_source),
        float(amount),
        float(time_sec),
        str(prediction_label),
        float(fraud_probability),
        str(risk_level),
        notes
    ))

    conn.commit()
    conn.close()
    return transaction_id


def insert_bulk_predictions(
    records: List[Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None
) -> int:
    """Insert bulk prediction records within an atomic transaction."""
    if not records:
        return 0

    conn = get_connection(db_path)
    cursor = conn.cursor()

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = []

    for r in records:
        tx_id = r.get("transaction_id", generate_transaction_id("TX-BLK-"))
        ts = r.get("timestamp", now_ts)
        params.append((
            str(tx_id),
            ts,
            str(r.get("prediction_source", "bulk")),
            float(r.get("amount", 0.0)),
            float(r.get("time_sec", 0.0)),
            str(r.get("prediction_label", "Legitimate")),
            float(r.get("fraud_probability", 0.0)),
            str(r.get("risk_level", "LOW")),
            r.get("notes")
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO transactions (
            transaction_id, timestamp, prediction_source, amount,
            time_sec, prediction_label, fraud_probability, risk_level, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, params)

    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


def get_history(
    search: Optional[str] = None,
    risk_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Union[str, Path]] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieve historical transaction predictions with search, filter, and pagination.

    Returns:
        Tuple of (records_list, total_count_matching_filters).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[Any] = []

    if search:
        where_clauses.append("(transaction_id LIKE ? OR notes LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    if risk_filter and risk_filter.upper() != "ALL":
        where_clauses.append("risk_level = ?")
        params.append(risk_filter.upper())

    if source_filter and source_filter.lower() != "all":
        where_clauses.append("prediction_source = ?")
        params.append(source_filter.lower())

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Count matching records
    cursor.execute(f"SELECT COUNT(*) FROM transactions {where_sql}", params)
    total_count = cursor.fetchone()[0]

    # Fetch paginated rows
    query_sql = f"""
        SELECT id, transaction_id, timestamp, prediction_source,
               amount, time_sec, prediction_label, fraud_probability, risk_level, notes
        FROM transactions
        {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    query_params = params + [int(limit), int(offset)]
    cursor.execute(query_sql, query_params)
    rows = cursor.fetchall()

    records = [dict(row) for row in rows]
    conn.close()
    return records, total_count


def get_history_stats(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Compute summary metrics from historical stored predictions."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_label = 'Fraudulent' THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN prediction_label = 'Legitimate' THEN 1 ELSE 0 END) as legit_count,
            AVG(fraud_probability) as avg_prob,
            SUM(amount) as total_volume
        FROM transactions
    """)
    row = cursor.fetchone()
    conn.close()

    total = row["total_predictions"] or 0
    fraud = row["fraud_count"] or 0
    legit = row["legit_count"] or 0
    avg_prob = float(row["avg_prob"] or 0.0)
    total_vol = float(row["total_volume"] or 0.0)

    return {
        "total_predictions": total,
        "fraud_count": fraud,
        "legit_count": legit,
        "fraud_rate": round((fraud / total * 100), 2) if total > 0 else 0.0,
        "avg_fraud_probability": round(avg_prob, 4),
        "total_volume_analyzed": round(total_vol, 2),
    }


def delete_prediction(transaction_id: str, db_path: Optional[Union[str, Path]] = None) -> bool:
    """Delete a specific transaction record."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE transaction_id = ?", (transaction_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def clear_history(db_path: Optional[Union[str, Path]] = None) -> int:
    """Clear all records from transaction history."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count
