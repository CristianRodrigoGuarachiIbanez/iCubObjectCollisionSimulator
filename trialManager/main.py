from featureRetrieval import TrialRetriever
from groundTruthRetrieval import GroundTruthRetriever
from h5Writer import H5GTWriter, H5Writer
from buildDirector import BuildDirector
from productBuilder import ProductBuilder
from typing import Dict, List, Tuple, TypeVar, Iterator, ValuesView, Generator, Any, Callable;
from numpy import ndarray, asarray
from dataStorage import DataStorage
from cython import declare, locals, char, array, bint
from concurrent.futures import ProcessPoolExecutor, Future, as_completed

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
def labelData(fileName: str, direction: str='ra', colRate: str=None) -> T:
    '''

    :param fileName: string file name
    :param colRate: sum, key
    :return: a sum of all collisions and non collisions separately
    '''
    gt: GroundTruthRetriever = GroundTruthRetriever();
    dictionary: Dict[str, int] = gt.groundTruthRetrievalOnTrial(fileName, direction=direction);
    if (colRate == 'sum'):
        return gt.sumValuesDict(dictionary);
    elif(colRate == 'sum2'):
        return  gt.sumValuesDictPerTrial(dictionary);
    elif(colRate == 'key'):
        return gt.getKeysDictAccordingCollision(dictionary);
    return dictionary

def saveLabelsInPickle(labelData: Callable, side: str ='la'):
    executor: ProcessPoolExecutor = ProcessPoolExecutor();
    commands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm"]  # "left_arm", "right_arm"];
    labels: List[Dict[str, int]] = [val for val in executor.map(labelData, commands)]
    Left_hand, Right_hand, Left_forearm, Right_forearm = None; #type: Dict[str, int]

    if(side =='la'):
        Left_hand: Dict[str, int] = labels[0]
        Right_hand: Dict[str, int] = labels[1] # labelData('right_hand', colRate='sum2');
        Left_forearm: Dict[str, int] = labels[2] # labelData('left_forearm', colRate='sum2');
        Right_forearm: Dict[str, int] = labels[3] #labelData('right_forearm', colRate='sum2');
    elif(side =='ra' ):
        Left_hand: Dict[str, int] = labels[0]
        Right_hand: Dict[str, int] = labels[1]  # labelData('right_hand', colRate='sum2');
        Left_forearm: Dict[str, int] = labels[2]  # labelData('left_forearm', colRate='sum2');
        Right_forearm: Dict[str, int] = labels[3]  # labelData('right_forearm', colRate='sum2');

    # ------------------------ save ground truth data
    writer: DataStorage = DataStorage()
    labelData: List[Dict[str, int]] = [ Left_hand, Right_hand, Left_forearm, Right_forearm]
    writer.storeData('data_right_side', data=labelData)


def recoverLabels(side: str) ->  List[Dict[str, int]]:

    # ---------------------------- PICKEL LOADER
    loader: DataStorage = DataStorage()
    i: int = 0;
    gen: Generator = loader.loadData(pickelFileName=side)
    # ------------------------------- recover data
    output: List[Dict[str, int]] = list();
    gtFile: Dict[str, int] = None;
    while True:
        try:
            gtFile = next(gen);
            output.append(gtFile);
        except  StopIteration as s:
            print('MAIN:', s)
            break
    return output

def main(filename: str = 'training_data.h5'):

    buildDirector: BuildDirector = BuildDirector();
    # --------------------------- WRITER
    writer: H5Writer = H5Writer(filename);
    # --------------------------- prepare image data
    biL: List[ndarray] = buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='la');
    biR: List[ndarray] = buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='ra');
    scL: List[ndarray] = buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='la')
    scR: List[ndarray] = buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='ra')

    # ------------------------------- prepare labels data
    labelLeftSide: List[Dict[str, int]] = recoverLabels('data_left_side')
    labelRightSide: List[Dict[str, int]] = recoverLabels('data_right_side')

    data: List[Any] = [biL, biR, scL, scR]
    for i in range(len(data)):
        if(i == 0 and len(data[i])> 0):
            data  = [data[i], labelLeftSide[0], labelLeftSide[1], labelLeftSide[2], labelLeftSide[3]]
            datasetnames: List[str] = ['binocular_features_left', 'gt_hl_left_hand', 'gt_hl_right_hand', 'gt_hl_right_forearm', 'gt_hl_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='binocular_left_side', datasetNames=datasetnames)
        elif(i ==1 and len(biR)>0):
            data = [data[i], labelRightSide[0], labelRightSide[1], labelRightSide[2], labelRightSide[3]]
            datasetnames: List[str] = ['binocular_features_right', 'gt_hr_left_hand', 'gt_hr_right_hand', 'gt_hr_right_forearm', 'gt_hr_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='binocular_right_side', datasetNames=datasetnames)
        elif(i ==2 and len(scL)>0):
            data = [data[i], labelLeftSide[0], labelLeftSide[1], labelLeftSide[2], labelLeftSide[3]]
            datasetnames: List[str] = ['scene_features_left', 'gt_hl_left_hand', 'gt_hl_right_hand', 'gt_hl_right_forearm', 'gt_hl_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='scene_left_side', datasetNames=datasetnames)
        elif (i ==3 and len(scR) > 0):
            data = [data[i], labelRightSide[0], labelRightSide[1], labelRightSide[2], labelRightSide[3]]
            datasetnames: List[str] = ['scene_features_right', 'gt_hr_left_hand', 'gt_hr_right_hand',  'gt_hr_right_forearm', 'gt_hr_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='scene_right_side', datasetNames=datasetnames)

    writer.closingH5PY()

if __name__ == '__main__':

    # executor: ProcessPoolExecutor = ProcessPoolExecutor();
    # commands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm"] #"left_arm", "right_arm"];
    # # featureCommands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm", "left_arm", "right_arm", "joint_coord", "object_coord", "head_coord", "hand_coord"]
    #
    # ----------------------------- retrieve Labels

    # # labels: List[Future] = [executor.submit(labelData,  command, 'sum2') for command in commands]
    # # features: List[Future] = [executor.submit(featureData, featureCommand) for featureCommand in featureCommands ]
    # # features: List[Dict[str, ndarray]] = [val for val in executor.map(featureData, featureCommands)]
    # # print(len(features))
    # print(len(labels))

    # ----------------------- COLLISIONS

    # left_hand:  Dict[str, ndarray] = features[0]#featureData('left_hand');
    # right_hand: Dict[str, ndarray] = features[1]#featureData('right_hand');
    # left_forearm: Dict[str, ndarray] = features[2]#featureData('left_forearm');
    # right_forearm: Dict[str, ndarray] = features[3]#featureData('right_forearm');
    # left_arm: Dict[str, ndarray] = features[4]
    # right_arm: Dict[str, ndarray] = features[5]
    #
    # head_coord: Dict[str, ndarray] = features[8]
    # hand_coord: Dict[str, ndarray] = features[9]
    # object_coord: Dict[str, ndarray] = features[7]
    # joints: Dict[str, ndarray] = features[6] #featureData('joint_coord');

    # --------------------- LABELS

    # print(len(joints));
    # print('LEFT HAND', len(left_hand))
    # print('RIGHT HAND', len(right_hand))
    # print('LEFT FOREARM', len(left_forearm))
    # print('RIGHT FOREARM',len(right_forearm))
    # print('RIGHT ARM',len(right_arm))
    # print('LEFT ARM', len(left_arm))
    #
    # print('HEAD COORD', len(head_coord))
    # print('HAND COORD', len(hand_coord))
    # print('JOINTS', len(joints))
    # print('OBJECT COORD', len(object_coord))

    # print('LEFT HAND', len(gt_hl_Right_hand))
    # print('RIGHT HAND', len(gt_hl_Right_hand))
    # print('LEFT FOREARM',len(gt_hl_Left_forearm))
    # print('RIGHT FOREARM',len(gt_hl_Right_forearm))
    # print('LEFT HAND', len(gt_hr_Right_hand))
    # print('RIGHT HAND', len(gt_hr_Right_hand))
    # print('LEFT FOREARM', len(gt_hr_Left_forearm))
    # print('RIGHT FOREARM', len(gt_hr_Right_forearm))
    main()


