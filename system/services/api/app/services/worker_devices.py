"""Worker-owned device discovery; API containers do not need PyTorch or a GPU."""

import json
import logging
import subprocess
import sys
import threading
from functools import lru_cache

from celery.signals import worker_ready, worker_shutdown
from redis import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)
PREFIX = "deeptrace:worker-devices:"
HEARTBEAT_SECONDS = 10
EXPIRY_SECONDS = 45
_stop = threading.Event()
_worker_key: str | None = None


@lru_cache
def device_registry() -> Redis:
    return Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)


def get_detection_devices() -> dict:
    """Advertise only devices supported by every live consumer of this queue."""
    records = []
    try:
        registry = device_registry()
        keys = list(registry.scan_iter(match=f"{PREFIX}{settings.detector_queue}:*"))
        for raw in registry.mget(keys) if keys else []:
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(record, dict) and record.get("cpu") is True:
                records.append(record)
    except Exception:
        logger.warning("暂时无法读取检测设备状态", exc_info=True)
    gpu = bool(records) and all(record.get("gpu") is True for record in records)
    names = sorted({record["gpu_name"] for record in records if isinstance(record.get("gpu_name"), str)})
    return {
        "worker_ready": bool(records),
        "cpu": bool(records),
        "gpu": gpu,
        "gpu_name": " / ".join(names) if gpu and names else None,
    }


def _probe_gpu() -> dict:
    # Probe in a separate process: initializing CUDA in the Celery parent would
    # make subsequent prefork pool replacements unsafe.
    code = (
        "import json,torch; available=torch.cuda.is_available(); "
        "print(json.dumps({'gpu':available,'gpu_name':"
        "torch.cuda.get_device_name(0) if available else None}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(completed.stdout)


def _heartbeat(key: str) -> None:
    record = {"cpu": True, "gpu": False, "gpu_name": None}
    try:
        registry = device_registry()
        try:
            registry.set(key, json.dumps(record), ex=EXPIRY_SECONDS)
        except Exception:
            logger.warning("检测服务启动时设备状态暂未发布，将继续重试", exc_info=True)
        try:
            record.update(_probe_gpu())
        except Exception:
            logger.warning("GPU 状态检查失败，仍可使用 CPU", exc_info=True)
        while not _stop.is_set():
            try:
                registry.set(key, json.dumps(record), ex=EXPIRY_SECONDS)
            except Exception:
                logger.warning("检测设备状态心跳暂时失败", exc_info=True)
            _stop.wait(HEARTBEAT_SECONDS)
    finally:
        try:
            device_registry().delete(key)
        except Exception:
            pass


@worker_ready.connect
def start_device_heartbeat(sender=None, **kwargs) -> None:
    global _worker_key
    _stop.clear()
    _worker_key = f"{PREFIX}{settings.detector_queue}:{sender.hostname}"
    threading.Thread(target=_heartbeat, args=(_worker_key,), daemon=True).start()


@worker_shutdown.connect
def stop_device_heartbeat(**kwargs) -> None:
    _stop.set()
    if _worker_key:
        try:
            device_registry().delete(_worker_key)
        except Exception:
            pass
