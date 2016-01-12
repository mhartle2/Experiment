import numpy as np
import math
from sklearn import datasets
iris = datasets.load_iris()

data = iris.data
targets = iris.target
combined = np.column_stack((data, targets))

for i in range(0, 50):
	np.random.shuffle(combined)

len_combined = len(combined)
seventy_percent = int(math.floor(len_combined * 0.7))
thirty_percent = len_combined - seventy_percent

training_set = np.zeros(shape=(seventy_percent, 5))
test_set = np.zeros(shape=(thirty_percent, 5))

for i in range(0, seventy_percent):
	training_set[i] = combined[i]

for i in range(0, thirty_percent):
	test_set[i] = combined[i + seventy_percent]


class HardCodedClassifier():
	def train(self, training_set):
		return

	def predict(self, test_set):
		return 0

	def accuracy(self, test_set):
		rate = 0
		for i in range(0, len(test_set)):
			if (test_set[i][4] == self.predict(test_set)):
				rate += 1
		return rate

hard_coded_classifier = HardCodedClassifier()
hard_coded_classifier.train(training_set)
hard_coded_classifier.predict(test_set)

number_right = hard_coded_classifier.accuracy(test_set)
accuracy = (float(number_right) / float(thirty_percent)) * 100

print(accuracy, end="")
print("% accuracy rate")