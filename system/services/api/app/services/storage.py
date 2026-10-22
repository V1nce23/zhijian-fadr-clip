from functools import lru_cache
from pathlib import Path
from typing import BinaryIO
import json

import boto3
from botocore.client import Config

from app.core.config import settings


class ObjectStorage:
    def __init__(self) -> None:
        client_options = {
            "aws_access_key_id": settings.s3_access_key,
            "aws_secret_access_key": settings.s3_secret_key,
            "region_name": settings.s3_region,
            "use_ssl": settings.s3_secure,
            "config": Config(signature_version="s3v4"),
        }
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            **client_options,
        )
        self.public_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_public_endpoint_url or settings.s3_endpoint_url,
            **client_options,
        )
        self.bucket = settings.s3_bucket

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self.client.exceptions.ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def upload(self, key: str, file_obj: BinaryIO, content_type: str) -> None:
        self.client.upload_fileobj(
            file_obj,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})

    def read_json(self, key: str) -> dict:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as body:
            return json.loads(body.read())

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def download(self, key: str, destination: Path) -> None:
        self.client.download_file(self.bucket, key, str(destination))

    def download_url(self, key: str, expires_seconds: int = 300) -> str:
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )


@lru_cache
def get_storage() -> ObjectStorage:
    return ObjectStorage()
