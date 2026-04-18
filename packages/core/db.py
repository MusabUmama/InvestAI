from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


_ENGINE = None
_SESSIONMAKER = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_database_url(), pool_pre_ping=True)
    return _ENGINE


def get_sessionmaker():
    global _SESSIONMAKER
    if _SESSIONMAKER is None:
        _SESSIONMAKER = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SESSIONMAKER


def db_session() -> Session:
    return get_sessionmaker()()
