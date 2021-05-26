from typing import List, Tuple, TypeVar, Any, Callable
from glob import glob
import logging

class FilesExtensionClassifier:

    def sortFilesAccordingToExtentions(self, filesList: List[str], extensionList: List[str]) ->List[List[str]]:
        assert (len(extensionList) > 0), "The Ending List ist empty"
        sortedList: List[List[str]] = list();
        for l in range(len(extensionList)):
            sortedList.append(self.__getCSVFiles(filesList, extensionList[l]));
        return sortedList

    def grabFilesWithExtensionCSV(self, fileList: List[str]) -> List[str]:
        return self.__getCSVFiles(fileList)

    def grabFilesWithExtensionH5(self, fileList: List[str], extension: str = '.h5') -> List[str]:
        return self.__getCSVFiles(fileList, extension);

    @staticmethod
    def __getCSVFiles(filesList: List[str], extension: str = ".csv") -> List[str]:
        csvFiles: List[str] = list()
        for files in filesList:
            if files.endswith(extension):
                csvFiles.append(files)
                # logging.info("the extension file {} could not be sorted properly".format(extension))
        return csvFiles;