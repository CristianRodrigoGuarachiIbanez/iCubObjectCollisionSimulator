from productBuilder import ProductBuilder;
from os import listdir, getcwd;
from filesExtensionClassifier import FilesExtensionClassifier
from pandas import DataFrame, to_numeric;
from typing import List, Tuple, TypeVar
from numpy import ndarray, float as Float;

class BuildDirector:
    """
       The Director is only responsible for executing the building steps in a
       particular sequence. It is helpful when producing products according to a
       specific order or configuration. Strictly speaking, the Director class is
       optional, since the client can control builders directly.
    """
    T: TypeVar = TypeVar("T", ndarray, DataFrame);
    def __init__(self) -> None:
        self._builder: ProductBuilder = None;

    @property
    def builder(self) -> ProductBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: ProductBuilder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._builder = builder

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def __recoverListOfDataFrame(self,) -> None:
        '''
        recovers a data frame
        :param dataName: string name of the data frame
        :return: a single data frame
        '''
        for file in listdir(getcwd()):
            if file.endswith(".zip"):
                #print("DIRECTOR:",file);
                self._builder.produceDataFrame(file);

    def __recoverImgArrayProduct(self) -> None:
        for file in listdir(getcwd()):
            if file.endswith(".zip"):
                #print("DIRECTOR:",file);
                self._builder.produceImgArray(file);

    # ------------------------- build the specific parts -----------------

    def buildIndividualDataFrame(self,  dataName: str, dtype: str = "numpy" ) -> T:
        self.__recoverListOfDataFrame()
        if(dtype =="numpy"):
            return self._builder.readSpecificDataFile(dataName).to_numpy();
        elif(dtype =="dataframe"):
            return self._builder.readSpecificDataFile(dataName)
        else:
            print('data type dtype is not valid. Introduce one of the following: numpy or dataframe')

    def buildImgArray(self, dataName:str) -> List[ndarray]:
        self.__recoverImgArrayProduct();
        if(dataName =='binocular_img'):
            return self._builder.readSpecificImgArray(dataName);
        elif(dataName == 'scene_img'):
            return self._builder.readSpecificImgArray(dataName);


    def buildSceneArray(self, dataName: str) -> List[ndarray]:
        #self.__recoverImgArrayProduct();
        return self._builder.readSpecificImgArray(dataName);



if __name__ == "__main__":
    buildDirector: BuildDirector = BuildDirector();
    csvBuilder: ProductBuilder = ProductBuilder();
    buildDirector.builder = csvBuilder;

    #left_arm: DataFrame = buildDirector.buildIndividualDataFrame("left_arm");
    head: ndarray = buildDirector.buildIndividualDataFrame("left_hand");
    bi: List[ndarray] = buildDirector.buildImgArray("binocular_img");
    #sc: List[ndarray] = buildDirector.buildSceneArray()
    print("LEFT HAND", len(buildDirector.buildIndividualDataFrame("left_hand")));
    print("IMG:", len(bi));

    #print(head)
