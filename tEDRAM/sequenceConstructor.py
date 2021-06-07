from numpy import ndarray, empty, int32, random, array
from typing import Generator, List
from pickle import load
from h5py import File

class SequenceConstructor:

    def __init__(self, rows: int, cols: int) -> None:
        '''
        create a matrix with a given number of rows and columns
        :param rows: integer, number of trials calculated from the total number of samples
        :param cols: integer, number of images in one image sequence
        '''
        self.__matrix: ndarray = empty((rows//10, cols), dtype=int);
        self._sequences: List[ndarray] = list()

    def getMatrix(self) -> ndarray:
        return self.__matrix

    def samples(self, features: ndarray, start: int, end: int) -> ndarray:
        listOfImgArraysSequences: List[ndarray] = list();
        currSequence: Generator = self.generateRandomIndexSequences(start, end);
        current: ndarray = None;
        if(len(self._sequences)!=0): self._sequences.clear();

        for i in range(start, end):
            current = next(currSequence);
            self._sequences.append(current)
            listOfImgArraysSequences.append(array(features[current, ...], dtype='float32'));
        return array(listOfImgArraysSequences, dtype='float32')

    def labels(self, labels: ndarray, start: int, end: int):
        assert(len(self._sequences) == abs(start-end)), ' the sequence list ist empty'
        listOfSequences: List[ndarray] = list()
        for i in range(len(self._sequences)):
            listOfSequences.append(array(labels[self._sequences[i], ...], dtype='int32'))
        return array(listOfSequences, dtype='int32')

    def generateRandomIndexSequences(self, start: int, end: int) -> Generator:
        '''
        returns the image sequences individually
        :param start: integer
        :param end: interger
        :return: generator with individual image sequences (one dime array)
        '''
        self.randomSequences()
        for i in range(start, end):
            yield self.__matrix[i]
    def randomSequences(self) -> None:
        self.sequences()
        # matrix should be reset hier
        random.shuffle(self.__matrix);
        #return matrix;

    def sequences(self) -> None:
        N: int = self.__matrix.shape[0]
        M: int = self.__matrix.shape[1]
        sequence = empty(M, dtype=int32)

        for i in range(N):
            sequence: ndarray = self.__indexSequence(sequence, i)
            for j in range(M):
                self.__matrix[i,j] = sequence[j]
        #return self.__matrix
    def __indexSequence(self, size: ndarray, index: int) -> ndarray:

        start: int = size.shape[0] * index;
        end: int = size.shape[0]*(index+1);
        counter: int = 0
        for i in range(start, end):
            size[counter] = i
            counter+=1
        return size
if __name__ == '__main__':
    s = SequenceConstructor(500, 10)
    print(s.getMatrix().shape)
    arr = File('training_data/training_data.h5', 'r')
    img: ndarray = arr['feature_data']['scene_data']
    print(img.shape)
    a = s.samples(img, start=0, end=10)

    with open('training_data/label_data.txt', 'rb') as file:
        data: ndarray = load(file)

        b = s.labels(data[0], start=0, end=10)

    print(a.shape)
    print()


    pass