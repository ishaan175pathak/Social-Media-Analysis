from pathlib import Path
import matplotlib.pyplot as plt
from pandas import DataFrame
import pyspark
from typing import List, Any


class DataVisualization:
    """
    This Module Runs Visualization on the preprocess dataframe and then save
    the results in the directory provided.
    """

    def __init__(self, /, outDir: Path | str | None = None) -> None:
        if outDir is not None:
            self.outDir: Path = Path(outDir)
        else:
            self.outDir: Path = Path("visual_ref")

        self.outDir.mkdir(exist_ok=True)

    def __sentiment_Distribution__(self, /, sp_df: pyspark.sql.DataFrame) -> None:
        df: DataFrame = sp_df.toPandas()

        plt.figure(figsize=(8, 5))
        plt.bar(df["target"], df["tweet_count"])
        plt.xticks([0, 1], ["Negative", "Positive"])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Tweet Count")
        plt.savefig(self.outDir / "sentiment_distribution.png", bbox_inches="tight")
        plt.show()

    def __tweet_per_hour__(self, /, sp_df: pyspark.sql.DataFrame) -> None:
        df: DataFrame = sp_df.toPandas()

        plt.figure(figsize=(10, 5))
        plt.plot(df["hour"], df["tweet_count"], marker="o")
        plt.title("Tweets Per Hour")
        plt.xlabel("Hour")
        plt.ylabel("Tweet Count")
        plt.xticks(range(0, 24))
        plt.savefig(self.outDir / "tweets_per_hour.png", bbox_inches="tight")
        plt.show()

    def __tweet_per_day__(self, /, sp_df: pyspark.sql.DataFrame) -> None:
        daily_pd: DataFrame = sp_df.toPandas()

        # create a readable date label
        daily_pd["date_label"] = (
            daily_pd["year"].astype(str) + "-" +
            daily_pd["month"].astype(str).str.zfill(2) + "-" +
            daily_pd["day"].astype(str).str.zfill(2)
        )

        plt.figure(figsize=(12, 5))
        plt.plot(daily_pd["date_label"], daily_pd["tweet_count"], marker="o")
        plt.title("Tweets Per Day")
        plt.xlabel("Date")
        plt.ylabel("Tweet Count")
        plt.xticks(rotation=45)
        plt.savefig(self.outDir / "tweets_per_day.png", bbox_inches="tight")
        plt.show()

    def __sentiment_over_time__(self, /, sp_df: pyspark.sql.DataFrame) -> None:
        df: DataFrame = sp_df.toPandas()

        df["date_label"] = (
            df["year"].astype(str) + "-" +
            df["month"].astype(str).str.zfill(2) + "-" +
            df["day"].astype(str).str.zfill(2)
        )

        pivot_df = df.pivot_table(
            index="date_label",
            columns="target",
            values="tweet_count",
            fill_value=0
        )

        plt.figure(figsize=(12, 5))
        pivot_df.plot(ax=plt.gca(), marker="o")
        plt.title("Sentiment Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Tweet Count")
        plt.xticks(rotation=45)
        plt.savefig(self.outDir / "sentiment_trend_over_time.png", bbox_inches="tight")
        plt.show()