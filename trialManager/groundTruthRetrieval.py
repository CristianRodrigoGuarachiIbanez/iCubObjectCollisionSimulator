from buildDirector import BuildDirector
from productBuilder import ProductBuilder
from numpy import ndarray, asarray, float32, delete, where
from typing import List, Any, Dict, Tuple, Generator
from cython import declare, locals, char, array, float, double, int as cint
import pyximport
#pyximport.install()
#pyximport.install(pyimport = True)
import counter

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
    def groundTruthRetrievalOnTrial(self, dataName: str, direction: str, trialSize: int = 10) -> Dict[str, int]:

        data: Generator = self.__arrayDataGenerator(dataName, direction=direction, stringArr=False)
        dataArr: ndarray = None; #next(data)

        counterF: int = 1;
        counterT: int = 1;
        currRow: int = None;
        finalArr: Dict[str, int] = dict();
        #for row in range(self.__dataLength):
        while True:
            try:
                dataArr = next(data)
                currRow = 1 if dataArr[self.__collisionRetrievalOnFrame(dataArr)].size > 0 else 0; #########
                finalArr["frame_" + str(counterF) + ', trial_' + str(counterT)] = currRow;
                counterF += 1;
            except StopIteration as s:
                print(s)
                break
            if (counterF >= trialSize + 1):
                counterF = 1;
                counterT += 1;
        return finalArr;

    @locals(dataName=char)
    def __arrayDataGenerator(self, dataName, direction: str, stringArr: bool = True) -> Generator:

        data: ndarray = self.buildCoordinateData(dataName, direction=direction)
        self.__dataLength: int = len(data)
        if (stringArr):
            return data;
        else:
            currArr: ndarray = None;
            row: str = None;

            dataArr: List[ndarray] = list();
            for row in range(len(data)):
                yield self.__deleteFirstElement(self.__numpyStringToFloat(data[row]));
                # currArr = self.__deleteFirstElement(self.__numpyStringToFloat(data[row]));
                #dataArr.append(currArr)
           # return asarray(dataArr);

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
    @locals(dictData=Dict)
    def sumValuesDict(dictData: Dict) -> str:
        noncollision: int = sum(value==0 for value in dictData.values());
        collision: int = sum(value==1 for value in dictData.values());
        return "no-collisions: {} | collisions: {}".format(noncollision, collision)
    @staticmethod
    @locals(dictData=Dict)
    def sumValuesDictPerTrial(dictData: Dict[str, int]) -> str:

        currKey: str = None
        previousKey: str = None;
        collisions: int = 0;
        noncollisions: int = 0;
        tempList: List[int] = list();
        for key, value in dictData.items():
            keys: List[str] = key.split(',');
            currKey= keys[1];
            if(currKey == previousKey):
                tempList.append(value);

            else:
                previousKey = currKey;
                if(len(tempList) != 0):
                    #collisions += 1 if 1 in tempList else 0;
                    collisions += tempList.count(1)
                    noncollisions += 1 if not 1 in tempList else 0;
                tempList.clear()
                tempList.append(value)

        return 'number of frames {} | number of trials {} | non-collisions:{} | collisions: {}'.format(len(dictData), len(dictData)//10, noncollisions, collisions);
    @staticmethod
    def cythonFunction(dictData: Dict[str, int]) -> str:
        return counter.getNonCollisionsOnly(dictData)

    @staticmethod
    @locals(dictData=array)
    def getKeysDictAccordingCollision(dictData: Dict, collision: bool = True) -> List[str]:
        if(collision):
            return [key for key, value in dictData.items() if value==1]
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
