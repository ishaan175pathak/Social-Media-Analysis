from __future__ import annotations

import os
import argparse
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple, Union, Literal
import numpy

# Unzipping and checking the dataset the dataset

class DataLoader:
    
    def __init__(self, fileName: str) -> None:
        """

        Extract and load the dataset in a Pandas DataFrame.

        Args:
            filename (zip file): Name of the Zip File.
        
        Returns: 
            None
        
        """

        if not str(fileName).endswith(".zip"):
            raise AttributeError(f"The File {fileName} is not a Zip File, please provide a ZipFile.")

        self.zipFile: Union[Path, str]= Path().parent.resolve() / str(fileName)
        self.datasetPath: Path = Path().parent.resolve() / "dataset"

    
    def  __extractFile__(self, fileName: str) -> None:
        """
            Extract the Zip File in the dataset folder

            Args:
                fileName (str): Name of the file to save as a CSV inside the data folders

            Returns:
                None 
        """

        self.dataset_dir: Path = self.datasetPath / fileName

        if not self.dataset_dir.exists():
            Path(self.dataset_dir).mkdir(exist_ok=True)
            with ZipFile(self.zipFile) as zfile:
                file_list: List[str] = zfile.namelist()

                for file in tqdm(iterable=file_list, total=len(file_list), desc="Extracting"):
                    zfile.extract(member=file, path=self.dataset_dir)
        else:
            print(f"File Already Exists with the File Name {fileName}")


    def __loadCSV__(self) -> pd.DataFrame:
        """
            Loading the CSV file in the memory

            Args:
                None

            Returns:
                DataFrame: Returns a DataFame of the CSV file
        """

        for fileName in self.dataset_dir.glob("*.csv"):
            csvFileName: Path = fileName

        return pd.read_csv(csvFileName, names=['target', 'id', 'date', 'flag', 'user', 'text'], sep=",", encoding='latin-1')


    def __getattr__(self, name):
        raise AttributeError(f"Error: {name} is not a valid method or property for this class")


    @property
    def printHead(self) -> pd.DataFrame:
        
        # extracting the csv file loading the pandas dataframe
        df: pd.DataFrame = self.__loadCSV__()
        return df.head()


    @property
    def printColumns(self) -> List[Any]:
        """
            Returns:
                List[Any]: Returns the List of Columns
        """
        return list(self.__loadCSV__().columns)
    
    
    @property
    def printColumnDtypes(self) -> Any:
        """
            Returns:
                List[str]: List of strings containing the data types of every folder
        """

        return self.__loadCSV__().dtypes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="Sentiment140.zip")
    args = parser.parse_args()

    fileName: str = str(args.file_name).removesuffix(".zip")

    try:    
        dl: DataLoader = DataLoader(args.file_name)
        dl.__extractFile__(fileName)
        dl.__loadCSV__()
        print(dl.printHead)
        print(dl.printColumns)
        print(dl.printColumnDtypes)
    except AttributeError as e:
        print(e)