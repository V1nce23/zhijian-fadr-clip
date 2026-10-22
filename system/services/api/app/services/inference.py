"""Cooperative cancellation and bounded media subprocesses."""

import subprocess
import time
from collections.abc import Callable


class InferenceCancelled(Exception):
    pass


class InferenceTimedOut(TimeoutError):
    pass


class InferenceControl:
    def __init__(
        self, timeout_seconds: float, cancelled: Callable[[], bool] = lambda: False
    ) -> None:
        self.deadline = time.monotonic() + timeout_seconds
        self.cancelled = cancelled

    def check(self) -> None:
        if self.cancelled():
            raise InferenceCancelled("任务已取消")
        if time.monotonic() >= self.deadline:
            raise InferenceTimedOut("检测器执行超时")


def run_command(
    command: list[str], *, timeout: float = 600, check: Callable[[], None] | None = None
) -> bytes:
    if check:
        check()
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        try:
            while True:
                if check:
                    check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(remaining, 0.25))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if check:
                check()
            if process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode, command, stdout, stderr
                )
            return stdout
        except BaseException:
            # Also reap ffmpeg on a Celery soft time limit or cancellation.
            process.kill()
            process.communicate()
            raise
