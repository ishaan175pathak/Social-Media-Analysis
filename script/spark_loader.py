from __future__ import annotations
from pyspark.sql import DataFrame, SparkSession


class SparkDataLoader:
    
    def __init__(
        self,
        app_name: str = "SocialMediaAnalysis",
        master: str = "local[*]",
    ) -> None:
        """
            Creates and manages a Spark session, and loads CSV data into Spark.

            Args:
                app_name (str): Name of the PySpark Application
                master (str): URL for the PySpark Application

            Returns:
                None
        """


        self.app_name = app_name
        self.master = master
        self.spark: SparkSession | None = None

    def create_session(self) -> SparkSession:
        """
        Create and return a Spark session.

        Returns:
            SparkSession: Returns an object of Spark Session
        """
        if self.spark is None:
            self.spark = (
                SparkSession.builder
                .appName(self.app_name)
                .master(self.master)
                .config("spark.hadoop.fs.defaultFS", "file:///")
                .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")
        return self.spark

    def load_csv(
        self,
        file_path: str,
        header: bool = True,
        infer_schema: bool = True,
        multiline: bool = True,
        escape: str = '"',
    ) -> DataFrame:
        """
        Load a CSV file into a Spark DataFrame.

        Args:
            File Path (str): Path to the CSV File
            Header (bool): Include the headers from the CSV File
            Infer Schema (bool): Infer the Schema from the CSV File
            Multiline (bool): Allow Multiple lines in the dataset
            Escape (str): String Patterns that are to be Ignored

        Returns:
            (DataFrame): DataFrame created through PySpark object
        """
        spark = self.create_session()

        df = (
            spark.read
            .option("header", str(header).lower())
            .option("inferSchema", str(infer_schema).lower())
            .option("multiline", str(multiline).lower())
            .option("escape", escape)
            .csv(file_path)
        )
        return df

    def stop_session(self) -> None:
        """
        Stop the Spark session if it exists.
        """
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
    