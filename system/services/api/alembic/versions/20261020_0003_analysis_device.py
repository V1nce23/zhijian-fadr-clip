"""Persist per-analysis CPU/GPU selection without guessing legacy devices."""

from alembic import op
import sqlalchemy as sa

revision = "20261020_0003"
down_revision = "20261013_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analyses", sa.Column("requested_device", sa.String(8), nullable=True))
    op.add_column("analyses", sa.Column("executed_device", sa.String(8), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "executed_device")
    op.drop_column("analyses", "requested_device")
