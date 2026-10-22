"""Track the hard execution deadline for abandoned analysis recovery."""

from alembic import op
from datetime import datetime, timedelta, timezone
import sqlalchemy as sa


revision = "20261013_0002"
down_revision = "20260722_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analyses",
        sa.Column("execution_deadline", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_analyses_execution_deadline", "analyses", ["execution_deadline"]
    )
    # Existing in-flight jobs have no deadline. Give them one deployment grace period.
    analyses = sa.table(
        "analyses",
        sa.column("status", sa.String),
        sa.column("execution_deadline", sa.DateTime(timezone=True)),
    )
    op.execute(
        analyses.update()
        .where(analyses.c.status.in_(["preprocessing", "running", "aggregating"]))
        .values(execution_deadline=datetime.now(timezone.utc) + timedelta(seconds=960))
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_execution_deadline", table_name="analyses")
    op.drop_column("analyses", "execution_deadline")
