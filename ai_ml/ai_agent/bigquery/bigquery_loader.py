from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import pandas as pd
from google.cloud.bigquery.table import TableReference
from loguru import logger

class BigQueryLoaderError(Exception):
    pass

class BigQueryLoader:
    """
    Class responsible for loading data into BigQuery.
    
    :param: project_id: The ID of the Google Cloud project.
    :param: dataset_id: The ID of the dataset.
    :param: location: The location of the dataset.
    :param: client (bigquery.Client, optional): The BigQuery client. Defaults to bigquery.Client().
    """
    def __init__(self, 
        project_id: str,
        location: str,
        client: bigquery.Client = bigquery.Client()
        ) -> None:
        self.project_id = project_id
        self.location = location
        self.client = client

    def table_exists(self, dataset_id: str, table_id: str):
        """ Check if table exists"""
        table_ref = self._get_destination_table(dataset_id, table_id)
        try:
            self.client.get_table(table_ref)
            return True
        except Exception as e:
            if isinstance(e, NotFound):
                return False
            else:
                raise e

    def _add_labels(self, table_ref: str, labels: str) -> None:
        """
        Adds labels to a BigQuery table.

        :param: table_ref: The destination table.
        :param: labels: The labels to be added.
        """
        table = self.client.get_table(table_ref)
        existing_labels = table.labels
        if existing_labels != labels:
            table.labels = labels
            table = self.client.update_table(table, ["labels"])
            logger.info(f"Added labels to {table.table_id}.")

    def _get_destination_table(self, dataset_id: str, table_id: str) -> str:
        """
        Builds the destination table name in BigQuery.
        
        :param:  table_id: The name of the table.
        Returns:
            str: The destination table name.
        """
        return TableReference.from_string(f"{self.project_id}.{dataset_id}.{table_id}")

    def load_dataframe(self, dataframe: pd.DataFrame, table_id: str, dataset_id: str, schema_fields: str = None, description: str = None, 
                        labels: dict = None, create_disposition: str = "CREATE_IF_NEEDED", write_disposition: str = "WRITE_APPEND", 
                        autodetect: str = True, schema_relax: str = None, time_partitioning: str = None, partition_field: str = None
        ) -> None:
        """
        Loads DataFrame data into BigQuery.

        :param: dataframe (pd.DataFrame): The DataFrame data to be loaded.
        :param: dataset_id: The ID of the dataset.
        :param: table_id: The name of the table.
        :param: schema_fields (optional): The schema fields for the data. Defaults to None.
        :param: description (optional): The table description. Defaults to None.
        :param: create_disposition (optional): The table creation disposition. Defaults to 'CREATE_IF_NEEDED'.
        :param: write_disposition: The write disposition for the job. Defaults to None.
        :param: autodetect (optional): Whether to automatically detect schema. Defaults to True.
        :param: schema_relax (optional): Whether to allow schema field addition. Defaults to False.
        :param: time_partitioning (optional): The type of time partitioning. Supported values are: `HOUR`,`DAY`,`MONTH`,`YEAR`
        :param: partition_field (optional): The field to partition the data by. Defaults to None.
        """
        table_ref = self._get_destination_table(dataset_id, table_id)
        
        job_config = bigquery.LoadJobConfig(
            schema=schema_fields,
            destination_table_description=description,
            create_disposition=create_disposition,
            write_disposition=write_disposition,
            autodetect=autodetect,
            schema_update_options=bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION if schema_relax else None,
            time_partitioning=bigquery.TimePartitioning(type_=time_partitioning.upper(), field=partition_field) if time_partitioning else None
        )
  
        job = self.client.load_table_from_dataframe(dataframe=dataframe, destination=table_ref, location=self.location, job_config=job_config)
        try:
            logger.info(f"Inserting data into the {table_id} in BigQuery!")
            job.result()
            logger.success("Successfully entered data")
            if labels:
                self._add_labels(table_ref, labels)
        except Exception as e:
            raise BigQueryLoaderError(f"Failed to insert data into BigQuery: {str(e)}")
        


    def bq_truncate(self, project_id, dataset_id, table_id):

        query = f"TRUNCATE TABLE `{project_id}.{dataset_id}.{table_id}`"
        query_job = self.client.query(query)
        query_job.result()