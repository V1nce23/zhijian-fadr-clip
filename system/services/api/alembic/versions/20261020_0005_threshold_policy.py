"""Save the selected decision rule without relabeling historical analyses."""

from alembic import op
import sqlalchemy as sa

revision = "20261020_0005"
down_revision = "20261020_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analyses", sa.Column("threshold_policy", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "threshold_policy")
