"""Exercise deletion against a real PostgreSQL writer holding the report lock."""

import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest
from sqlalchemy import event, select

from app.core.database import SessionLocal, engine
from app.models import Analysis
from app.services.evidence_gallery import gallery_prefix
from app.services.inference import InferenceCancelled
from app.services.storage import get_storage
from test_live_visualization import running


@pytest.mark.skipif(engine.dialect.name != "postgresql", reason="Requires PostgreSQL row locks")
@pytest.mark.parametrize("batch", [False, True])
def test_delete_waits_for_writer_and_removes_its_last_assets(client, batch):
    identity, run_id, reporter = running(client)
    storage = client.app.dependency_overrides[get_storage]()
    deleting = threading.Event()
    writer_thread = threading.get_ident()

    def observe_delete_lock(connection, cursor, statement, parameters, context, many):
        if threading.get_ident() != writer_thread and "FOR UPDATE" in statement:
            deleting.set()

    def delete():
        return (client.post("/api/v1/analyses/batch-delete", json={"analysis_ids": [identity]})
                if batch else client.delete(f"/api/v1/analyses/{identity}"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with SessionLocal() as db:
            analysis = db.scalar(select(Analysis).where(Analysis.id == uuid.UUID(identity)).with_for_update())
            event.listen(engine, "before_cursor_execute", observe_delete_lock)
            try:
                result = pool.submit(delete)
                assert deleting.wait(5), "Deletion must request the same analysis row lock"
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.2)
                # Simulate the last progress/asset update already in flight.
                analysis.progress = 58
                storage.upload(gallery_prefix(identity, run_id) + "last-frame.png", io.BytesIO(b"last asset"), "image/png")
                db.commit()
            finally:
                event.remove(engine, "before_cursor_execute", observe_delete_lock)
        assert result.result(timeout=5).status_code == 200
    assert client.get(f"/api/v1/analyses/{identity}").status_code == 404
    assert not any(key.startswith(gallery_prefix(identity)) for key in storage.objects)
    with pytest.raises(InferenceCancelled):
        reporter("reporting")
