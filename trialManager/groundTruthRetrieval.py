from buildDirector import BuildDirector
from productBuilder import ProductBuilder
from numpy import ndarray, asarray, float32, delete, where
from typing import List, Any, Dict, Tuple
from cython import declare, locals, char, array, float, double, int as cint


class GroundTruthRetriever(BuildDirector):
    prodBuilder = declare(ProductBuilder)
    def __init__(self):

        super(GroundTruthRetriever, self).__init__();
        self.prodBuilder: ProductBuilder = ProductBuilder();
        self.__loadBuildDirector()

    def __loadBuildDirector(self) -> None:
        self.builder = self.prodBuilder;

    @locals(frameArr=ndarray)
    def __collisionRetrievalOnFrame(self, frameArr: ndarray) -> ndarray:

        return where(frameArr!=0.) # collect index values bigger than 0.

    @locals(data=char, trialSize=cint)
    def groundTruthRetrievalOnTrial(self, dataName: str, trialSize: int = 15) -> Dict[str, int]:
        data: array = self.__arrayDataGenerator(dataName, stringArr=False)
        counterF: int = 1;
        counterT: int = 1;
        currRow: int = None;
        finalArr: Dict[str, int] = dict();
        for row in range(len(data)):
            currRow = 1 if data[self.__collisionRetrievalOnFrame(data[row])].size > 0 else 0 ;
            finalArr["frame_" + str(counterF) + ', trial_' + str(counterT)] = currRow;
            counterF += 1;
            if (counterF >= trialSize + 1):
                counterF = 1;
                counterT += 1;

        return finalArr;

    @locals(dataName=char)
    def __arrayDataGenerator(self, dataName, stringArr: bool = True) -> ndarray:
        data: ndarray = self.buildIndividualDataFrame(dataName)
        if (stringArr):
            return data;
        else:
            currArr: ndarray = None;
            row: str = None;

            dataArr: List[ndarray] = list();
            for row in range(len(data)):
                currArr = self.__deleteFirstElement(self.__numpyStringToFloat(data[row]));
                dataArr.append(currArr)
            return asarray(dataArr);

    @staticmethod
    @locals(oneDimArray=ndarray)
    def __numpyStringToFloat(oneDimArray: ndarray) -> ndarray:
        elem: str = None;
        num: str = None;
        floatingArr: ndarray = None;
        try:
            return oneDimArray.astype(float32);
        except ValueError as e:
            try:
                for elem in range(len(oneDimArray)):
                    floatingArr = asarray([float(num) for num in oneDimArray[elem].split(",")], dtype=float32);
            except Exception as e:
                print(e);

        return floatingArr

    @staticmethod
    @locals(oneDimArray=ndarray)
    def __deleteFirstElement(oneDimArray: ndarray) -> ndarray:
        return delete(oneDimArray, 0, None);

    @staticmethod
    @locals(dictData=array)
    def sumValuesDict(dictData: Dict) -> str:
        noncollision: int = sum(value==0 for value in dictData.values());
        collision: int = sum(value==1 for value in dictData.values());
        return "no-collisions: {} | collisions: {}".format(noncollision, collision)
    @staticmethod
    @locals(dictData=array)
    def getKeysDict(dictData: Dict, collision: bool = True) -> List[str]:
        if(collision):
            return [key  for key, value in dictData.items() if value==1]
        elif(collision is False):
            return [key for key, value in dictData.items() if value==0 ]


if __name__ == '__main__':
    # gtData: GroundTruthRetriever = GroundTruthRetriever();
    # left_hand: Dict[str, int] = gtData.groundTruthRetrievalOnTrial('left_hand');
    # right_hand: Dict[str, int] = gtData.groundTruthRetrievalOnTrial('right_hand')
    # collisionRate: str =  gtData.sumValuesDict(right_hand)
    # collisionItems: List[str] = gtData.getKeysDict(right_hand)
    #
    # print(left_hand, end="\n");
    # print(right_hand);
    # print(collisionRate)
    # print(collisionItems)
    pass
