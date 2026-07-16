from typing import Any
from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, split, col
from pathlib import Path
from typing import List


class DataAnalysis:
    """
    This module performs essential Explanatory Data Analysis such as
    Statistical Analysis on Spark DataFrame
    """

    def __init__(self, sparkObject: DataFrame) -> None:
        self.df: DataFrame = sparkObject

    def __sentimentDistribution__(self) -> DataFrame:
        print("Title: Analyzing Sentiment Distribution")
        return (
            self.df
            .groupBy("target")
            .count()
            .withColumnRenamed("count", "tweet_count")
        )

    def __tweetsPerHours__(self) -> DataFrame:
        print("Title: Analyzing Tweets per Hour")
        return (
            self.df
            .groupBy("hour")
            .count()
            .withColumnRenamed("count", "tweet_count")
            .orderBy("hour")
        )

    def __tweetsPerDay__(self) -> DataFrame:
        print("Title: Analyzing Tweets per Day")
        return (
            self.df
            .groupBy("year", "month", "day")
            .count()
            .withColumnRenamed("count", "tweet_count")
            .orderBy("year", "month", "day")
        )

    def __sentimentOverTime__(self) -> DataFrame:
        print("Title: Analyzing Sentiment over Time")
        return (
            self.df
            .groupBy("year", "month", "day", "target")
            .count()
            .withColumnRenamed("count", "tweet_count")
            .orderBy("year", "month", "day", "target")
        )

    def __topWords__(self) -> DataFrame:
        print("Title: Analyzing Top Words")
        return (
            self.df
            .withColumn("words", explode(split(col("text"), " ")))
            .groupBy("words")
            .count()
            .orderBy("count", ascending=False)
        )