from __future__ import annotations

import json
import logging
import os
from typing import Any


_LOGGER: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger("cs336_basics")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "bpe.log"), encoding="utf-8")
        fmt = logging.Formatter(
            "time=%(asctime)s level=%(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    _LOGGER = logger
    return logger


def _fmt_value(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (dict, list, tuple)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return repr(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            return json.dumps(v.decode("utf-8"), ensure_ascii=False)
        except Exception:
            return repr(v)
    return str(v)


def info_kvs(*pairs: Any, **kvs: Any) -> None:
    logger = _get_logger()
    items: dict[str, Any] = {}
    if pairs:
        assert len(pairs) % 2 == 0, "pairs must be key,value repeated"
        for i in range(0, len(pairs), 2):
            items[str(pairs[i])] = pairs[i + 1]
    if kvs:
        items.update(kvs)
    try:
        msg = json.dumps(items, ensure_ascii=False)
    except Exception:
        # Fallback to stringifying values if JSON serialization fails
        safe_items = {k: _fmt_value(v) for k, v in items.items()}
        msg = json.dumps(safe_items, ensure_ascii=False)
    logger.info(msg)
