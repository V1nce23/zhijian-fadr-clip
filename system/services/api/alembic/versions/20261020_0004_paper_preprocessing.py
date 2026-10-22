"""Record raw versus already aligned input without relabeling historical reports."""

from alembic import op
import sqlalchemy as sa

revision = "20261020_0004"
down_revision = "20261020_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analyses", sa.Column("input_face_mode", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "input_face_mode")
