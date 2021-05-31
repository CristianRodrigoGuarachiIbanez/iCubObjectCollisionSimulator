
from typing import List, Tuple, Generator, TypeVar
from random import shuffle
from numpy import ndarray, array, asarray

class SequenceGenerator:

	def __init__(self, numOfTrials: int, sequenceLength: int) -> None:
		self.__numOfTrials: int = numOfTrials;
		self.__sequenceLength: int = sequenceLength;

	def getSequenceList(self) -> List[List[int]]:
		return self.__indexSequenceList();

	def samples(self, features: ndarray, start: int, end: int) -> ndarray:
		listOfSequnces: List[ndarray] = list();
		counter: int = end
		generator: Generator = self.sequencesOfIndices(start, end);
		sequence: ndarray = None;
		while(counter > 0):
			try:
				sequence = array(features[next(generator), ...], dtype='float32')
				listOfSequnces.append(sequence);
			except Exception as e:
				print(e);
			counter -=1;
		return array(listOfSequnces);

	def sequencesOfIndices(self, start: int, end: int) -> Generator:
		sequences: List[List[int]] = self.__indexSequenceList();
		shuffle(sequences);
		sampleSequences = sequences[start:end]
		for i in range(len(sampleSequences)):
			yield sampleSequences[i];

	def __indexSequenceList(self):
		sampleLists: List[List[int]] = list();
		sequences : Generator = self.__sequenceGenerator(self.__numOfTrials, self.__sequenceLength);
		for sequence in sequences:
			sampleLists.append(sequence);
		return sampleLists;

	@staticmethod
	def __sequenceGenerator(numOfTrials: int,sequenceLength: int) -> Generator:
		assert(isinstance(numOfTrials, int)), " number of trials soll ein Intergerwert sein"
		for item in range(numOfTrials):
			yield list(range(sequenceLength*item, sequenceLength*(item+1)));
if __name__ == '__main__':

	# dataset_size: int = 70000 # anzahl an trials
	# trial_size: int = 10 # sequence
	# trialSample_size: int = dataset_size //trial_size
	# #size: List[int] = [trial_size] * trialSample_size
	# sampleSize: int = 100
	#
	# indices: SequenceGenerator = SequenceGenerator(trialSample_size, trial_size);
	# start: int = 0;
	# end: int = 10;
	# gen: Generator = indices.sequencesOfIndices(start, end);
	#
	# for g in gen:
	# 	print(g)
	pass












