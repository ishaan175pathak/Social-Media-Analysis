from spark_loader import SparkDataLoader
from spark_preprocessing import SparkPreprocessor
from dataAnalysis import DataAnalysis
from dataVisualization import DataVisualization
from argparse import ArgumentParser
from pathlib import Path
from load_dataset import DataLoader
from typing import Any, List, Union
import pandas as pd 
import pyspark.sql
from tqdm import tqdm
import shutil


def main() -> None:
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--dir_name", default="Sentiment140")

    args: Any = parser.parse_args()

    dataset_dir: Path = Path("./dataset") / args.dir_name

    fileName: Path | None = None

    for csvFile in Path(dataset_dir).glob("*.csv"):
        fileName = csvFile

    if fileName is None:
        raise FileNotFoundError("No CSV file found")

    file_path = fileName.resolve().as_uri()

    print("Using file:", file_path)

    text_column = "text"
    timestamp_column = "date"

    loader = SparkDataLoader(app_name="SocialMediaAnalysis")
    df = loader.load_csv(
        file_path=file_path,
        header=False,
        infer_schema=True
    )

    df = df.toDF("target", "id", "date", "flag", "user", "text")

    print("\n=== Raw Schema ===")
    df.printSchema()

    print("\n=== Raw Sample ===")
    df.show(5, truncate=False)

    preprocessor = SparkPreprocessor(
    df=df,
    text_column="text",
    timestamp_column="date",
    )

    selected_columns = ["target", "id", "date", "user", "text"]

    processed_df = (
        preprocessor
        .select_columns(selected_columns)
        .drop_nulls()
        .drop_duplicates()
        .clean_text()
        .remove_empty_text()
        .format_timestamp(input_format="MMM dd HH:mm:ss z yyyy")
        .add_time_features()
        .map_target_labels()
        .get_dataframe()
    )

    print("\n=== Processed Schema ===")
    processed_df.printSchema()

    print("\n=== Processed Sample ===")
    processed_df.show(10, truncate=False)

    print("\n=== Distinct Targets ===")
    processed_df.select("target").distinct().show()

    print("\n=== Time Features Sample ===")
    processed_df.select("date", "year", "month", "day", "hour").show(10, truncate=False)

    print(f"\nTotal records after preprocessing: {processed_df.count()}")

    print("\n=== Performing Data Analysis ===")

    da_ob: DataAnalysis = DataAnalysis(processed_df)    

    da_methods: List[Any] = [da_ob.__sentimentDistribution__, da_ob.__tweetsPerDay__, da_ob.__tweetsPerHours__,\
               da_ob.__sentimentOverTime__, da_ob.__topWords__]
    
    da_res: List[pyspark.sql.DataFrame] = []
  
    for method in tqdm(da_methods, desc="Running Exections"):
        df: pyspark.sql.DataFrame = method()
        da_res.append(df)
        print(df.show(20))

    print("\n=== Completed execution: Data Analysis ===")

    print("\n=== Performing Data Visualization ===")

    print("\n=== Performing Data Visualization ===")

    # deleting the visual refs folder if it exists

    if Path("visual_ref").exists() and Path("visual_ref").is_dir():
        shutil.rmtree("visual_ref")

    dv_ob: DataVisualization = DataVisualization()

    dv_methods: List[Any] = [
        dv_ob.__sentiment_Distribution__,
        dv_ob.__tweet_per_day__,
        dv_ob.__tweet_per_hour__,
        dv_ob.__sentiment_over_time__
    ]

    # da_res order:
    # 0 -> sentimentDistribution
    # 1 -> tweetsPerDay
    # 2 -> tweetsPerHours
    # 3 -> sentimentOverTime
    # 4 -> topWords

    for method, df_result in tqdm(
        zip(dv_methods, da_res[:4]),
        desc="Running Execution"
    ):
        method(df_result)

    print("\n=== Completed Execution: Data Visualization ===")

    print("\n=== Preparing Subset for Model Training ===")

    # export full cleaned dataset from Spark as CSV shards

    processed_df\
    .select("target", "id", "date", "user", "text", "year", "month", "day", "hour")\
    .write\
    .mode("overwrite")\
    .option("header", True)\
    .csv("exports/cleaned_tweets_csv")


    print("Saved cleaned dataset to exports/cleaned_tweets_csv")

    print("\n=== Training the Model ===")

    loader.stop_session()


if __name__ == "__main__":
    main()