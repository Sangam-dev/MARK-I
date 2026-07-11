"""SQLite-backed structured memory using aiosqlite."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .base import AbstractMemoryBackend
from core.exceptions import MemoryError


class StructuredMemory(AbstractMemoryBackend):
    """SQLite-backed structured memory for conversations, facts, and tasks."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    @property
    def name(self) -> str:
        return "structured"

    async def initialize(self) -> None:
        """Create tables and enable WAL mode."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrency
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.commit()

        # Create tables
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                due_at TEXT,
                status TEXT DEFAULT 'pending',
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_session
                ON interactions(session_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_session_key
                ON facts(session_id, key);
            CREATE INDEX IF NOT EXISTS idx_tasks_session_status
                ON tasks(session_id, status);
        """)
        await self._conn.commit()

    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        """Store generic content. Delegates to store_interaction."""
        session_id = metadata.get("session_id", "default")
        role = metadata.get("role", "user")
        return await self.store_interaction(session_id, role, content, metadata)

    async def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve recent interactions matching query (simple text search)."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        # Simple text search in content
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, role, content, timestamp, metadata
            FROM interactions
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def delete(self, record_id: str) -> bool:
        """Delete a record by ID from any table."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        # Try each table
        for table in ("interactions", "facts", "tasks"):
            cursor = await self._conn.execute(
                f"DELETE FROM {table} WHERE id = ?", (record_id,)
            )
            if cursor.rowcount > 0:
                await self._conn.commit()
                return True
        return False

    async def clear_session(self, session_id: str) -> int:
        """Clear all records for a session across all tables."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        total = 0
        for table in ("interactions", "facts", "tasks"):
            cursor = await self._conn.execute(
                f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
            )
            total += cursor.rowcount
        await self._conn.commit()
        return total

    async def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            if not self._conn:
                return False
            await self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # --- Interaction methods ---

    async def store_interaction(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a conversation interaction."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})

        await self._conn.execute(
            """
            INSERT INTO interactions (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (record_id, session_id, role, content, timestamp, meta_json),
        )
        await self._conn.commit()
        return record_id

    async def get_recent_interactions(
        self, session_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent interactions for a session."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            """
            SELECT id, session_id, role, content, timestamp, metadata
            FROM interactions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    # --- Fact methods ---

    async def store_fact(
        self, key: str, value: str, session_id: str
    ) -> str:
        """Store or update a fact."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        now = datetime.utcnow().isoformat()

        # Check if fact exists
        cursor = await self._conn.execute(
            "SELECT id FROM facts WHERE session_id = ? AND key = ?",
            (session_id, key),
        )
        existing = await cursor.fetchone()

        if existing:
            # Update existing
            await self._conn.execute(
                """
                UPDATE facts SET value = ?, updated_at = ?
                WHERE id = ?
                """,
                (value, now, existing["id"]),
            )
            record_id = existing["id"]
        else:
            # Insert new
            record_id = str(uuid.uuid4())
            await self._conn.execute(
                """
                INSERT INTO facts (id, key, value, session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (record_id, key, value, session_id, now, now),
            )
        await self._conn.commit()
        return record_id

    async def get_fact(self, key: str, session_id: str) -> str | None:
        """Get a fact by key."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            "SELECT value FROM facts WHERE session_id = ? AND key = ?",
            (session_id, key),
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def get_all_facts(self, session_id: str) -> list[dict[str, Any]]:
        """Get all facts for a session."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            "SELECT id, key, value, created_at, updated_at FROM facts WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # --- Task methods ---

    async def store_task(
        self,
        description: str,
        due_at: str | None,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Store a task."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        record_id = task_id or str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})

        await self._conn.execute(
            """
            INSERT INTO tasks (id, description, due_at, status, session_id, created_at, metadata)
            VALUES (?, ?, ?, 'pending', ?, ?, ?)
            """,
            (record_id, description, due_at, session_id, created_at, meta_json),
        )
        await self._conn.commit()
        return record_id

    async def get_pending_tasks(self, session_id: str) -> list[dict[str, Any]]:
        """Get pending tasks for a session."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            """
            SELECT id, description, due_at, status, session_id, created_at, metadata
            FROM tasks
            WHERE session_id = ? AND status = 'pending'
            ORDER BY due_at ASC NULLS LAST, created_at ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "description": row["description"],
                "due_at": row["due_at"],
                "status": row["status"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def get_all_pending_tasks(self) -> list[dict[str, Any]]:
        """Get all pending tasks across all sessions."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            """
            SELECT id, description, due_at, status, session_id, created_at, metadata
            FROM tasks
            WHERE status = 'pending'
            ORDER BY due_at ASC NULLS LAST, created_at ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "description": row["description"],
                "due_at": row["due_at"],
                "status": row["status"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        if not self._conn:
            raise MemoryError("Database not initialized")

        cursor = await self._conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id)
        )
        await self._conn.commit()
        return cursor.rowcount > 0