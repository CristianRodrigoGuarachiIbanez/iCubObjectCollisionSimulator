from featureRetrieval import TrialRetriever
from groundTruthRetrieval import GroundTruthRetriever
from cython import declare, locals
from typing import Dict, List, Tuple, TypeVar
from numpy import ndarray
from cython import declare, locals, char, array, bint

@locals(fileName=char)
def featureData(fileName: str)-> Dict[str, ndarray] :
    trial: TrialRetriever = TrialRetriever();
    # recover cvs data in trials
    return trial.callTrialDataArrAsDict(fileName);
@locals(fileName=char)
def imgData(fileName: str) ->  Dict[str, ndarray]:
    trial: TrialRetriever = TrialRetriever();
    # recover img array data as trials
    # ojo always binocular img array has to be called first in order for scene to get the data
    if(fileName =='binocular_img'):
        return trial.callImgDataArr(fileName);
    elif(fileName =='scene_img'):
        return trial.callImgDataArr(fileName);

T: TypeVar = TypeVar('T',Dict[str, int], List[str], str)
@locals(fileName=char, colRate=bint)
def labelData(fileName: str, colRate: str='sum') -> T:
    '''

    :param fileName: string file name
    :param colRate: sum, key
    :return: a sum of all collisions and non collisions separately
    '''
    gt: GroundTruthRetriever = GroundTruthRetriever();
    dictionary: Dict[str, int] = gt.groundTruthRetrievalOnTrial(fileName);
    if (colRate == 'sum'):
        return gt.sumValuesDict(dictionary);
    elif(colRate == 'key'):
        return gt.getKeysDict(dictionary);
    return dictionary


if __name__ == '__main__':

    pass



