from typing import Sequence

import pandas as pd
from airflow.models import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


class S3ToPostgresOperator(BaseOperator):

    """
    Custom operator to transfer a file from S3 to a Postgres table.
    Assumes the file is CSV (no header) and loads it using pandas.
    """

    template_fields: Sequence[str] = ("bucket", "key", "table")

    def __init__(
        self,
        bucket: str,
        key: str,
        table: str,
        postgres_conn_id: str = "neon_db_conn",
        aws_conn_id: str = "aws_default",
        if_exists: str = "replace",  # <-- ajout param pour contrôler le mode
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bucket = bucket
        self.key = key
        self.table = table
        self.postgres_conn_id = postgres_conn_id
        self.aws_conn_id = aws_conn_id
        self.if_exists = if_exists

    def execute(self, context):
        # Télécharger le fichier depuis S3
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        returned_filename = s3_hook.download_file(
            key=self.key, bucket_name=self.bucket, local_path="/tmp"
        )

        # Charger en DataFrame
        df_file = pd.read_csv(returned_filename, header=None)

        # Connexion Postgres
        postgres_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        engine = postgres_hook.get_sqlalchemy_engine()

        # Charger les données en base
        df_file.to_sql(self.table, engine, if_exists=self.if_exists, index=False)
