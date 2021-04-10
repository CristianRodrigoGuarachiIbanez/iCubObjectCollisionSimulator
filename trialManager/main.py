from featureRetrieval import TrialRetriever
from groundTruthRetrieval import GroundTruthRetriever
from cython import declare, locals
from typing import Dict, List, Tuple
from numpy import ndarray

@locals()
def main()-> int:
    trial: TrialRetriever = TrialRetriever();

    # recover cvs data in trials
    left_hand: Dict[str, ndarray] = trial.callTrialDataArrAsDict('left_hand');
    right_hand: Dict[str, ndarray] = trial.callTrialDataArrAsDict('right_hand')
    left_forearm: Dict[str, ndarray] = trial.callTrialDataArrAsDict('left_forearm');
    right_forearm: Dict[str, ndarray] = trial.callTrialDataArrAsDict('right_forearm');
    left_arm: Dict[str, ndarray] = trial.callTrialDataArrAsDict('left_arm');


    # recover img array data as trials
    # ojo always binocular img array has to be called first in order for scene to get the data
    binoImgArr: Dict[str, ndarray] = trial.callImgDataArr('binocular_img')
    sceneImgArr: Dict[str, ndarray] = trial.callImgDataArr('scene_img')

    print(len(left_hand))
    print(len(right_hand))
    print(len(binoImgArr))
    print(len(sceneImgArr))