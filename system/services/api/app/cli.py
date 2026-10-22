import argparse

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models import User, UserRole


def seed_admin(email: str | None, password: str | None, name: str) -> None:
    if not email or not password:
        raise SystemExit("必须通过参数或环境变量提供管理员邮箱和密码")
    if len(password) < 10:
        raise SystemExit("管理员密码至少需要 10 个字符")

    normalized_email = email.lower()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == normalized_email))
        if user:
            user.name = name
            user.role = UserRole.ADMIN
            user.is_active = True
            db.commit()
            print(f"管理员已存在：{normalized_email}")
            return
        db.add(
            User(
                email=normalized_email,
                name=name,
                password_hash=hash_password(password),
                role=UserRole.ADMIN,
            )
        )
        db.commit()
        print(f"管理员已创建：{normalized_email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="智鉴管理命令")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser("seed-admin", help="创建初始管理员")
    seed_parser.add_argument("--email", default=settings.admin_email)
    seed_parser.add_argument("--password", default=settings.admin_password)
    seed_parser.add_argument("--name", default=settings.admin_name)
    args = parser.parse_args()

    if args.command == "seed-admin":
        seed_admin(args.email, args.password, args.name)


if __name__ == "__main__":
    main()
