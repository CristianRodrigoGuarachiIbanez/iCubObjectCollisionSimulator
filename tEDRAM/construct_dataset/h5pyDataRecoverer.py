from numpy import ndarray, asarray, array, zeros, ones
from typing import List, Dict, Tuple, Any, Generator, TypeVar
from sequenceGenerator import SequenceGenerator
from imageEdition import ImageEditor
from h5py import File, Group

class H5PYDataRetriever:
    T: TypeVar = TypeVar('T', List[Dict], List[ndarray])

    def __init__(self) -> None:
        pass

    def getDataFile(self, filename: str) -> Generator:
        with File(filename, "r") as self._file:
            datasets: List[str] = list(self._file.keys())
            print(datasets)
            data: Any = None;
            for dataset in range(len(datasets)):
                data = self._file[datasets[dataset]][:];
                self._file.close()
                yield asarray(data)
    @staticmethod
    def writerDataIntoH5PY( filename: str, groupname:str, datasetname: List[str], data: T) -> None:
        group: Group = None
        datasetLength: int = len(datasetname);
        dataLength: int = len(data);

        with File(filename, 'w') as file:
            group = file.create_group(groupname)
            assert(datasetLength==dataLength), 'The number of datasets and data content are not equal'
            for i in range(dataLength):
                group.create_dataset(datasetname[i], data= data[i]);



if __name__ == '__main__':
    fileName: str = 'binocular_perception.h5'
    fileName2: str = 'scene_records.h5'
    opener: H5PYDataRetriever = H5PYDataRetriever();
    # batch ist hier die gesamte Stichprobegröße
    gen: Generator = opener.getDataFile(fileName);
    img: ndarray = next(gen);
    gen2: Generator = opener.getDataFile(fileName2);
    scene: ndarray = array(next(gen2));
    print("SCENE SHAPE:",scene.shape);
    # ------------------------------ create a h5 file
    newname: str = 'newfile'
    groupname: str = 'cameras'
    datasets: List[str] = ['binocularCameras', 'sceneCamera']
    data: List[ndarray] = [img, scene]
    opener.writerDataIntoH5PY(newname, groupname, datasets, data)

    # ------------------------------ picture edition
    edition: ImageEditor = ImageEditor()
    imgList: List[ndarray] = list()
    for i in range(len(scene)):
        imgList.append(edition.editImagArray(scene[i], equa_method='clahe', scale=80))
    print(len(imgList))
    for i in range(len(imgList)):
        print(edition.showImage(imgList[i]))
        print(imgList[i].shape)
    # ------------------------------ sequence contruction ------------------
    dataset_size: int = 51; # Anzahl an Bilder/Frames im gesammten Datensatz
    trial_size: int = 10; # Anzahl an Frames im Trial
    trialSample_size: int = dataset_size//trial_size # Anzahl an Trials aus dem gesammten Frames-Anzahl

    start, end = 0,5 #type: int, int # BATCH SIZE

    sequences: SequenceGenerator = SequenceGenerator(trialSample_size, trial_size);
    I: ndarray = sequences.samples(scene, start, end)
    print(I.shape)






