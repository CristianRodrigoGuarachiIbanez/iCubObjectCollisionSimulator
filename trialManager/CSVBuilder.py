from CSVDataProduct import CSVDataCollector
from builderInterface import BuilderInterface
from zipfile import ZipFile
from typing import List
from trialsManager import TrialManager
from filesExtensionClassifier import FilesExtensionClassifier
from pandas import read_csv, DataFrame
import logging

class CSVBuilder(BuilderInterface):

    listOfFileEndings: List[str] = [".csv", ".h5", "png"];
    def __init__(self) -> None:

        self.__csvCollector: CSVDataCollector = CSVDataCollector();
        self.__reset();

    def __reset(self) -> None:

        self._csvDataCollector: CSVDataCollector = CSVDataCollector();

    @property
    def product(self) -> CSVDataCollector:
        """
        Concrete Builders are supposed to provide their own methods for retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another product.
        That's why it's a usual practice to call the reset method at the end of
        the `getProduct` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        product: CSVDataCollector = self._csvDataCollector;
        self.__reset();
        return product;

    def produceRef(self) -> None:
        pass

    def __produceDataFrameFromCSV(self, zipFileName: str) -> None:
        fileClassifier: FilesExtensionClassifier = FilesExtensionClassifier()
        with ZipFile(zipFileName, "r") as zipClass:
            print("succefully imported")
            # zip.printdir()
            zipFiles: List[str] = zipClass.namelist();  # names of the files in the Zip
            sortedZipFiles: List[List[str]] = fileClassifier.sortFilesAccordingToExtentions(zipFiles, self.listOfFileEndings);  # zip file names sorting according to extension

            trialsManager: TrialManager = TrialManager(zipClass, sortedZipFiles)
            #  ITERATE OVER EVERY SINGLE ELEMENT OF THE SORTED ZIP FILES
            for zipFiles in sortedZipFiles:
                for fileWithExtension in zipFiles:

                    if (fileWithExtension.endswith(".csv") and not fileWithExtension.startswith("gt_")):  # select only the certain extension
                        print(fileWithExtension)
                        # GETS THE CONTENT OF FILES AND SAVES THEM ACCORDING TO EXTRACTED FILE ELEMENTS
                        try:
                            trialsManager.copyCSVData(fileWithExtension, trialsManager.generateDataFrame(fileWithExtension));
                        except Exception as e:
                            print(e)
                            print(fileWithExtension)

    def produceDataFrame(self, zipFileName: str) -> None:
        self.__produceDataFrameFromCSV(zipFileName);
        print(self.__csvCollector.getArmLeft())
        print(self.__csvCollector.getArmRight())



if __name__=="__main__":
    path: str = "iCubObjectCollisionSimulator/DGP_iCubSimulator/trials/ra_cube_trial6.zip";
    file: str = "gt_right_hand.csv";
    builder: CSVBuilder = CSVBuilder();
    builder.produceDataFrame(path)
