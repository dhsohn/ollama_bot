"""DFT 계산 결과 인덱스 — 별도 SQLite DB(dft.db)로 구조화된 메타데이터 관리.

ORCA 출력 파일을 파싱하여 dft_calculations 테이블에 저장하고,
다양한 필터 조건으로 SQL 검색을 수행한다.
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import partial
from pathlib import Path
from typing import Any

import aiosqlite

from core.dft_discovery import discover_orca_targets
from core.logging_setup import get_logger
from core.orca_parser import parse_orca_output

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS dft_calculations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path        TEXT    NOT NULL UNIQUE,
    file_hash          TEXT    NOT NULL,
    mtime              REAL    NOT NULL,
    calc_type          TEXT    NOT NULL,
    method             TEXT    NOT NULL,
    basis_set          TEXT    NOT NULL DEFAULT '',
    charge             INTEGER DEFAULT 0,
    multiplicity       INTEGER DEFAULT 1,
    formula            TEXT    NOT NULL DEFAULT '',
    n_atoms            INTEGER DEFAULT 0,
    energy_hartree     REAL,
    energy_ev          REAL,
    energy_kcalmol     REAL,
    opt_converged      INTEGER,
    has_imaginary_freq INTEGER,
    lowest_freq_cm1    REAL,
    enthalpy           REAL,
    gibbs_energy       REAL,
    wall_time_seconds  INTEGER,
    status             TEXT    NOT NULL DEFAULT 'completed',
    indexed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dft_method  ON dft_calculations(method);
CREATE INDEX IF NOT EXISTS idx_dft_formula ON dft_calculations(formula);
CREATE INDEX IF NOT EXISTS idx_dft_status  ON dft_calculations(status);
CREATE INDEX IF NOT EXISTS idx_dft_energy  ON dft_calculations(energy_hartree);
CREATE INDEX IF NOT EXISTS idx_dft_mtime   ON dft_calculations(mtime);
"""


class DFTIndex:
    """DFT 계산 결과의 구조화된 인덱스를 관리한다."""

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None
        self._db_path: str = ""
        self._write_lock = asyncio.Lock()
        self._logger = get_logger("dft_index")

    async def initialize(self, db_path: str) -> None:
        """데이터베이스를 열고 스키마를 생성한다."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        self._logger.info("dft_index_initialized", db_path=db_path)

    async def close(self) -> None:
        """데이터베이스 연결을 닫는다."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DFTIndex가 아직 초기화되지 않았습니다.")
        return self._db

    # ------------------------------------------------------------------
    # 인덱싱
    # ------------------------------------------------------------------

    async def index_calculations(
        self,
        kb_dirs: list[str],
        *,
        max_file_size_mb: int = 64,
    ) -> dict[str, Any]:
        """kb_dirs에서 ORCA 출력 파일을 스캔하여 인덱싱한다.

        file_hash 기반 증분 인덱싱: 변경된 파일만 재파싱한다.

        Returns:
            {"indexed": int, "skipped": int, "removed": int, "failed": int, "total": int}
        """
        db = self._require_db()

        # 기존 인덱스 로드
        async with db.execute(
            "SELECT source_path, file_hash FROM dft_calculations"
        ) as cursor:
            existing: dict[str, str] = {
                row["source_path"]: row["file_hash"] async for row in cursor
            }

        # 파일 탐색 (동기 파일 I/O를 스레드 풀로 위임)
        max_bytes = max_file_size_mb * 1024 * 1024
        discovered: dict[str, str] = {}  # path → hash

        def _discover_and_hash() -> dict[str, str]:
            result: dict[str, str] = {}
            for kb_dir in kb_dirs:
                kb_path = Path(kb_dir)
                if not kb_path.is_dir():
                    continue
                for fpath in discover_orca_targets(kb_path, max_bytes=max_bytes, logger=self._logger):
                    spath = str(fpath)
                    h = hashlib.sha256()
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    result[spath] = h.hexdigest()[:16]
            return result

        discovered = await asyncio.to_thread(_discover_and_hash)

        # 존재하지 않는 디렉토리 경고 (로거는 메인 스레드에서 호출)
        for kb_dir in kb_dirs:
            if not Path(kb_dir).is_dir():
                self._logger.warning("dft_kb_dir_not_found", path=kb_dir)

        # 변경 감지
        to_index = {
            p: h for p, h in discovered.items()
            if existing.get(p) != h
        }
        to_remove = set(existing) - set(discovered)

        indexed = 0
        failed = 0
        removed = 0

        # 삭제된 파일 제거
        async with self._write_lock:
            for rpath in to_remove:
                await db.execute(
                    "DELETE FROM dft_calculations WHERE source_path = ?", (rpath,)
                )
                removed += 1

            # 신규/변경 파일 인덱싱
            for source_path in to_index:
                try:
                    result = await asyncio.to_thread(parse_orca_output, source_path)
                    await self._upsert(db, result)
                    indexed += 1
                except Exception as exc:
                    self._logger.warning(
                        "dft_parse_failed", path=source_path, error=str(exc)
                    )
                    failed += 1

            await db.commit()

        total = await self._count()
        self._logger.info(
            "dft_index_complete",
            indexed=indexed,
            skipped=len(discovered) - len(to_index),
            removed=removed,
            failed=failed,
            total=total,
        )
        return {
            "indexed": indexed,
            "skipped": len(discovered) - len(to_index),
            "removed": removed,
            "failed": failed,
            "total": total,
        }

    async def upsert_single(self, file_path: str) -> bool:
        """단일 파일을 파싱하여 upsert한다. 성공 시 True."""
        db = self._require_db()
        try:
            result = await asyncio.to_thread(parse_orca_output, file_path)
            async with self._write_lock:
                await self._upsert(db, result)
                await db.commit()
            return True
        except Exception as exc:
            self._logger.warning("dft_upsert_failed", path=file_path, error=str(exc))
            return False

    async def _upsert(self, db: aiosqlite.Connection, r: Any) -> None:
        """OrcaResult를 dft_calculations에 upsert한다."""
        await db.execute(
            """INSERT INTO dft_calculations (
                source_path, file_hash, mtime, calc_type, method, basis_set,
                charge, multiplicity, formula, n_atoms,
                energy_hartree, energy_ev, energy_kcalmol,
                opt_converged, has_imaginary_freq, lowest_freq_cm1,
                enthalpy, gibbs_energy, wall_time_seconds, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_hash=excluded.file_hash,
                mtime=excluded.mtime,
                calc_type=excluded.calc_type,
                method=excluded.method,
                basis_set=excluded.basis_set,
                charge=excluded.charge,
                multiplicity=excluded.multiplicity,
                formula=excluded.formula,
                n_atoms=excluded.n_atoms,
                energy_hartree=excluded.energy_hartree,
                energy_ev=excluded.energy_ev,
                energy_kcalmol=excluded.energy_kcalmol,
                opt_converged=excluded.opt_converged,
                has_imaginary_freq=excluded.has_imaginary_freq,
                lowest_freq_cm1=excluded.lowest_freq_cm1,
                enthalpy=excluded.enthalpy,
                gibbs_energy=excluded.gibbs_energy,
                wall_time_seconds=excluded.wall_time_seconds,
                status=excluded.status,
                indexed_at=CURRENT_TIMESTAMP
            """,
            (
                r.source_path, r.file_hash, r.mtime, r.calc_type, r.method, r.basis_set,
                r.charge, r.multiplicity, r.formula, r.n_atoms,
                r.energy_hartree, r.energy_ev, r.energy_kcalmol,
                1 if r.opt_converged is True else (0 if r.opt_converged is False else None),
                1 if r.has_imaginary_freq is True else (0 if r.has_imaginary_freq is False else None),
                r.lowest_freq_cm1,
                r.enthalpy, r.gibbs_energy, r.wall_time_seconds, r.status,
            ),
        )

    async def _count(self) -> int:
        db = self._require_db()
        async with db.execute("SELECT COUNT(*) FROM dft_calculations") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # 쿼리 메서드
    # ------------------------------------------------------------------

    async def query(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """동적 필터 조건으로 계산 결과를 검색한다.

        지원 필터:
            method, basis_set, calc_type, status, formula,
            energy_min, energy_max, opt_converged, has_imaginary_freq
        """
        db = self._require_db()
        conditions: list[str] = []
        params: list[Any] = []

        for col in ("method", "basis_set", "calc_type", "status", "formula"):
            if value := filters.get(col):
                conditions.append(f"{col} = ?")
                params.append(value)

        if "method_like" in filters:
            conditions.append("method LIKE ?")
            params.append(f"%{filters['method_like']}%")

        if "formula_like" in filters:
            conditions.append("formula LIKE ?")
            params.append(f"%{filters['formula_like']}%")

        if "energy_min" in filters:
            conditions.append("energy_hartree >= ?")
            params.append(filters["energy_min"])
        if "energy_max" in filters:
            conditions.append("energy_hartree <= ?")
            params.append(filters["energy_max"])

        if "opt_converged" in filters:
            conditions.append("opt_converged = ?")
            params.append(1 if filters["opt_converged"] else 0)

        if "has_imaginary_freq" in filters:
            conditions.append("has_imaginary_freq = ?")
            params.append(1 if filters["has_imaginary_freq"] else 0)

        where = " AND ".join(conditions) if conditions else "1=1"
        limit = min(int(filters.get("limit", 50)), 200)
        order = filters.get("order_by", "mtime DESC")

        # order_by 화이트리스트
        allowed_orders = {
            "mtime DESC", "mtime ASC",
            "energy_hartree ASC", "energy_hartree DESC",
            "indexed_at DESC", "formula ASC",
        }
        if order not in allowed_orders:
            order = "mtime DESC"

        sql = f"SELECT * FROM dft_calculations WHERE {where} ORDER BY {order} LIMIT ?"
        params.append(limit)

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """전체 인덱스 통계를 반환한다."""
        db = self._require_db()
        stats: dict[str, Any] = {}

        async with db.execute("SELECT COUNT(*) FROM dft_calculations") as cur:
            row = await cur.fetchone()
            stats["total"] = row[0] if row else 0

        async with db.execute(
            "SELECT status, COUNT(*) as cnt FROM dft_calculations GROUP BY status"
        ) as cur:
            stats["by_status"] = {row["status"]: row["cnt"] async for row in cur}

        async with db.execute(
            "SELECT method, COUNT(*) as cnt FROM dft_calculations "
            "GROUP BY method ORDER BY cnt DESC LIMIT 10"
        ) as cur:
            stats["by_method"] = {row["method"]: row["cnt"] async for row in cur}

        async with db.execute(
            "SELECT calc_type, COUNT(*) as cnt FROM dft_calculations "
            "GROUP BY calc_type ORDER BY cnt DESC"
        ) as cur:
            stats["by_calc_type"] = {row["calc_type"]: row["cnt"] async for row in cur}

        async with db.execute(
            "SELECT formula, COUNT(*) as cnt FROM dft_calculations "
            "GROUP BY formula ORDER BY cnt DESC LIMIT 10"
        ) as cur:
            stats["top_formulas"] = {row["formula"]: row["cnt"] async for row in cur}

        return stats

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """가장 최근에 수정된 계산 결과를 반환한다."""
        return await self.query({"order_by": "mtime DESC", "limit": limit})

    async def get_lowest_energy(
        self,
        formula: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """에너지가 가장 낮은 계산 결과를 반환한다."""
        filters: dict[str, Any] = {
            "order_by": "energy_hartree ASC",
            "limit": limit,
        }
        if formula:
            filters["formula"] = formula
        return await self.query(filters)

    async def search_by_formula(self, formula: str) -> list[dict[str, Any]]:
        """화학식으로 검색한다 (정확 일치 + LIKE)."""
        exact = await self.query({"formula": formula})
        if exact:
            return exact
        return await self.query({"formula_like": formula})

    async def get_for_comparison(
        self,
        formula: str | None = None,
        method: str | None = None,
    ) -> list[dict[str, Any]]:
        """비교 분석용 데이터를 반환한다. 에너지 기준 정렬."""
        filters: dict[str, Any] = {
            "order_by": "energy_hartree ASC",
            "limit": 50,
        }
        if formula:
            filters["formula"] = formula
        if method:
            filters["method"] = method
        return await self.query(filters)
