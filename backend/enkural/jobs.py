"""Background job queue.

One worker thread executes jobs sequentially (the GPU is shared), publishing
stage / progress updates that the UI polls.  Stages follow the user-facing
sequence:

    Uploading… → Analyzing voice… → Extracting speech… → Converting voice…
    → Enhancing audio… → Finalizing… → "Voice conversion complete."
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from typing import Callable, Dict, Optional

from . import store

log = logging.getLogger(__name__)

STAGES = [
    ("upload", "Uploading..."),
    ("analyze", "Analyzing voice..."),
    ("extract", "Extracting speech..."),
    ("convert", "Converting voice..."),
    ("enhance", "Enhancing audio..."),
    ("finalize", "Finalizing..."),
]
STAGE_INDEX = {k: i for i, (k, _) in enumerate(STAGES)}
STAGE_LABEL = dict(STAGES)
# rough share of total time per stage for the overall progress bar
STAGE_WEIGHT = {"upload": 0.02, "analyze": 0.10, "extract": 0.10, "convert": 0.68, "enhance": 0.06, "finalize": 0.04}


class JobCancelled(Exception):
    pass


class JobContext:
    """Handle passed to job functions for reporting progress."""

    def __init__(self, job: Dict):
        self.job = job
        self._cancel = threading.Event()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def stage(self, key: str, detail: str = "", label: Optional[str] = None) -> None:
        self.check()
        self.job["stage"] = key
        self.job["stage_label"] = label or STAGE_LABEL.get(key, key)
        self.job["detail"] = detail
        self.job["progress"] = round(self._overall(key, 0.0) * 100)
        store.save_job(self.job)

    def progress(self, fraction: float, detail: Optional[str] = None) -> None:
        self.check()
        key = self.job.get("stage") or "convert"
        self.job["progress"] = round(self._overall(key, fraction) * 100)
        if detail:
            self.job["detail"] = detail
        store.save_job(self.job)

    @staticmethod
    def _overall(key: str, fraction: float) -> float:
        done = 0.0
        for k, _ in STAGES:
            if k == key:
                break
            done += STAGE_WEIGHT[k]
        return min(1.0, done + STAGE_WEIGHT.get(key, 0.0) * max(0.0, min(1.0, fraction)))


class JobManager:
    def __init__(self) -> None:
        self._q: "queue.Queue[tuple[str, Callable[[JobContext], Dict]]]" = queue.Queue()
        self._contexts: Dict[str, JobContext] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="enkural-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()
            # anything left "running" from a previous crash is marked failed
            for j in store.list_jobs():
                if j.get("status") in ("queued", "running"):
                    j["status"] = "error"
                    j["error"] = "Interrupted by a server restart."
                    store.save_job(j)

    def submit(self, job: Dict, fn: Callable[[JobContext], Dict]) -> Dict:
        job.setdefault("status", "queued")
        job.setdefault("stage", "upload")
        job.setdefault("stage_label", STAGE_LABEL["upload"])
        job.setdefault("progress", 0)
        job.setdefault("detail", "")
        job.setdefault("created_at", store.now_iso())
        store.save_job(job)
        ctx = JobContext(job)
        with self._lock:
            self._contexts[job["id"]] = ctx
        self._q.put((job["id"], fn))
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ctx = self._contexts.get(job_id)
        if ctx:
            ctx._cancel.set()
            return True
        return False

    def _run(self) -> None:
        while True:
            job_id, fn = self._q.get()
            with self._lock:
                ctx = self._contexts.get(job_id)
            if ctx is None:
                continue
            job = ctx.job
            if ctx.cancelled():
                job["status"] = "cancelled"
                store.save_job(job)
                continue
            job["status"] = "running"
            job["started_at"] = store.now_iso()
            store.save_job(job)
            try:
                result = fn(ctx)
                job.update(result or {})
                job["status"] = "done"
                job["progress"] = 100
                job["stage"] = "done"
                job["stage_label"] = job.get("done_label", "Voice conversion complete.")
                job["detail"] = ""
            except JobCancelled:
                job["status"] = "cancelled"
                job["detail"] = "Cancelled."
            except Exception as e:  # noqa: BLE001
                log.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"[:500]
            finally:
                job["finished_at"] = store.now_iso()
                store.save_job(job)
                with self._lock:
                    self._contexts.pop(job_id, None)


manager = JobManager()
