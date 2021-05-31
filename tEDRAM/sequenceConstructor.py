from numpy import ndarray, empty, int32, random, array
from typing import Generator, List

class SequenceConstructor:

    def __init__(self, rows: int, cols: int) -> None:
        '''
        create a matrix with a given number of rows and columns
        :param rows: integer, number of trials calculated from the total number of samples
        :param cols: integer, number of images in one image sequence
        '''
        self.__matrix: ndarray = empty((rows//10, cols), dtype=int);

    def getMatrix(self) -> ndarray:
        return self.__matrix

    def samples(self, features: ndarray, start: int, end: int) -> ndarray:
        listOfImgArraysSequences: List[ndarray] = list();
        currSequence: Generator = self.generateRandomIndexSequences(start, end);
        for i in range(start, end):
            listOfImgArraysSequences.append(array(features[next(currSequence), ...], dtype='float32'));
        return array(listOfImgArraysSequences, dtype='float32')

    def generateRandomIndexSequences(self, start: int, end: int) -> Generator:
        '''
        returns the image sequences individually
        :param start: integer
        :param end: interger
        :return: generator with individual image sequences
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
    # s = SequenceConstructor(50, 10)
    #
    # s.randomSequences()
    # o = s.getMatrix()
    # print(o)
    pass

