from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class SparkPreprocessor:
    """
    Handles Spark-based preprocessing for social media text data.
    """

    def __init__(self, df: DataFrame, text_column: str, timestamp_column: str | None = None) -> None:
        self.df = df
        self.text_column = text_column
        self.timestamp_column = timestamp_column

    def select_columns(self, columns: list[str]) -> "SparkPreprocessor":
        self.df = self.df.select(*columns)
        return self

    def drop_nulls(self) -> "SparkPreprocessor":
        required_columns = [self.text_column]
        if self.timestamp_column:
            required_columns.append(self.timestamp_column)

        self.df = self.df.dropna(subset=required_columns)
        return self

    def drop_duplicates(self) -> "SparkPreprocessor":
        self.df = self.df.dropDuplicates()
        return self

    def clean_text(self) -> "SparkPreprocessor":
        """
        Light cleaning only.
        Keeps transformer-friendly structure while removing obvious noise.
        """
        cleaned_col = F.col(self.text_column)

        cleaned_col = F.regexp_replace(cleaned_col, r"http\S+|www\S+", "")
        cleaned_col = F.regexp_replace(cleaned_col, r"@\w+", "")
        cleaned_col = F.regexp_replace(cleaned_col, r"#", "")
        cleaned_col = F.regexp_replace(cleaned_col, r"[^a-zA-Z0-9\s]", " ")
        cleaned_col = F.regexp_replace(cleaned_col, r"\s+", " ")
        cleaned_col = F.trim(cleaned_col)

        self.df = self.df.withColumn(self.text_column, cleaned_col)
        return self

    def map_target_labels(self) -> "SparkPreprocessor":
        self.df = self.df.withColumn(
            "target",
            F.when(F.col("target") == 4, 1).otherwise(0)
        )
        return self

    def remove_empty_text(self) -> "SparkPreprocessor":
        self.df = self.df.filter(F.length(F.col(self.text_column)) > 0)
        return self

    def format_timestamp(self, input_format: str | None = None) -> "SparkPreprocessor":
        """
        Convert timestamp column to Spark timestamp type if available.
        """
        
        if self.timestamp_column:
            # Remove weekday prefix like "Mon "
            cleaned_ts = F.regexp_replace(F.col(self.timestamp_column), r"^[A-Za-z]{3}\s+", "")

            if input_format:
                self.df = self.df.withColumn(
                    self.timestamp_column,
                    F.to_timestamp(cleaned_ts, input_format),
                )
            else:
                self.df = self.df.withColumn(
                    self.timestamp_column,
                    F.to_timestamp(cleaned_ts),
                )
        
        return self

    def add_time_features(self) -> "SparkPreprocessor":
        """
        Add useful time-based columns for trend analysis.
        """
        if self.timestamp_column:
            self.df = (
                self.df
                .withColumn("year", F.year(F.col(self.timestamp_column)))
                .withColumn("month", F.month(F.col(self.timestamp_column)))
                .withColumn("day", F.dayofmonth(F.col(self.timestamp_column)))
                .withColumn("hour", F.hour(F.col(self.timestamp_column)))
            )
        return self

    def get_dataframe(self) -> DataFrame:
        return self.df