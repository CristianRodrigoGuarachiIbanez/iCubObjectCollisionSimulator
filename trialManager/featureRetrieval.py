from typing import List, Tuple, Any, Dict
from buildDirector import BuildDirector
from productBuilder import ProductBuilder
from numpy import ndarray, float as Float, hstack, asarray, delete, double, float32
import cython

class TrialRetriever:

    builDirector = cython.declare(BuildDirector)
    csvBuilder = cython.declare(ProductBuilder)

    def __init__(self):
        self._buildDirector = BuildDirector();
        self._dataBuilder = ProductBuilder();
        self.__loadBuilDirector();

    def __loadBuilDirector(self) -> None:
        self._buildDirector.builder = self._dataBuilder;

    @cython.locals(dataName=cython.char, trialSize=cython.int)
    def callImgDataArr(self, dataName: str, trialSize: int = 15) -> Dict[str,ndarray]:
        data: List[ndarray] = None;
        if(dataName == 'binocular_img'):
            data = self._buildDirector.buildImgArray(dataName);
        elif(dataName =='scene_img'):
            data = self._buildDirector.buildImgArray(dataName);
        else:
            print('binocular_img or scene_img')
        counterF: int = 1;
        counterT: int = 1;
        imgArrayDict: Dict[str, ndarray] = dict();
        currArr: ndarray = None;
        for arr in data:
            for imgArr in arr:
                imgArrayDict["frame_" + str(counterF)+', trial_'+str(counterT)] = imgArr;
                counterF+=1;
                if(counterF>=trialSize+1):
                    counterF =1;
                    counterT +=1;
        return imgArrayDict

    @cython.locals(dataName=cython.char)
    def callTrialDataArr(self, dataName: str, stringArr: bool = True) -> ndarray:
        data: ndarray = self._buildDirector.buildIndividualDataFrame(dataName)
        if(stringArr):
            return data;
        else:
            currArr: ndarray = None;
            row: str = None;

            dataArr: List[ndarray] = list();
            for row in range(len(data)):
                currArr = self.__deleteFirstElement(self.__numpyStringToFloat(data[row]));
                dataArr.append(currArr)
            return asarray(dataArr);

    @cython.locals(data=cython.array, trialSize=cython.int)
    def callTrialDataArrAsDict(self, dataName: str, trialSize: int =15) -> Dict[str, ndarray]:
        data: ndarray = self._buildDirector.buildIndividualDataFrame(dataName);
        counterF: int = 1;
        counterT: int = 1;
        currRow: ndarray = None;
        finalArr: Dict[str, ndarray] = dict();
        for row in range(len(data)):
            #currRow: ndarray = insert(data[row], 0, counter, axis=0);
            #finalArr.append(currRow);
            currRow = self.__deleteFirstElement(self.__numpyStringToFloat(data[row]));
            finalArr["frame_" + str(counterF)+', trial_'+str(counterT)] = currRow;
            counterF+=1;
            if(counterF>=trialSize+1):
                counterF =1;
                counterT +=1;

        return finalArr;
    @staticmethod
    @cython.locals(oneDimArray=ndarray)
    def __numpyStringToFloat(oneDimArray: ndarray) -> ndarray:
        elem: str = None;
        num: str = None;
        floatingArr: ndarray = None;
        try:
            return oneDimArray.astype(Float);
        except ValueError as e:
            try:
                for elem in range(len(oneDimArray)):
                    floatingArr = asarray([float(num) for num in oneDimArray[elem].split(",")], dtype=float32);
            except Exception as e:
                print(e);

        return floatingArr

    @staticmethod
    @cython.locals(oneDimArray=ndarray)
    def __deleteFirstElement(oneDimArray: ndarray) -> ndarray:
        return delete(oneDimArray, 0, None);


if __name__ == '__main__':
    # trial: TrialRetriever = TrialRetriever();
    # # example data as dict
    # #left_hand: ndarray = trial.callTrialDataArr('left_hand');
    # #right_hand: ndarray = trial.callTrialDataArr('right_hand')
    #
    # # recover cvs data in trials
    # left_hand: Dict[str, ndarray] = trial.callTrialDataArrAsDict('left_hand');
    # right_hand: Dict[str, ndarray] = trial.callTrialDataArrAsDict('right_hand')
    #
    # # recover img array data as trials
    # # ojo always binocular img array has to be called first in order for scene to get the data
    # binoImgArr: Dict[str, ndarray] = trial.callImgDataArr('binocular_img')
    # sceneImgArr: Dict[str, ndarray] = trial.callImgDataArr('scene_img')
    #
    # print(len(left_hand))
    # print(len(right_hand))
    # print(len(binoImgArr))
    # print(len(sceneImgArr))
    pass







