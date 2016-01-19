import numpy as np
import math
#######################################
iris = open("iris.csv")

len_iris = 150
combined = np.zeros(shape=(len_iris, 5))
incrementer = 0
iris.readline()
for line in iris:
        line_to_append = line.split(",")
        for x in range(0, 5):
                combined[incrementer][x] = line_to_append[x]
        incrementer += 1


for i in range(0, 50):
        np.random.shuffle(combined)

len_combined = len(combined)
seventy_percent = int(math.floor(len_combined * .7))
thirty_percent = len_combined - seventy_percent

training_set = np.zeros(shape=(seventy_percent, 5))
test_set = np.zeros(shape=(thirty_percent, 5))

for i in range(0, seventy_percent):
        training_set[i] = combined[i]

for i in range(0, thirty_percent):
        test_set[i] = combined[i + seventy_percent]


class HardCodedClassifier():
        def calculate_z(self, mean, stdev, value):
                numerator = value - mean
                z = numerator / stdev
                return z

        def calculate_mean(self, training_set, column_number):
                total = 0
                for i in range(0, 105):
                        total += training_set[i][column_number]
                        mean = total / 105
                        return mean
                
        def calculate_std(self, training_set, column_number):
                std = np.array(training_set).std(0)
                return std[column_number]

        

        def z_filler(self, training_set, column_number, zeroed, mean, stdev, range_size):
                for i in range(0, range_size):
                        zeroed[i] = self.calculate_z(mean, stdev, training_set[i][column_number])

        
        def euclid(self, final_zero, target_data, range_size):
                euclidean_distance = np.zeros(shape=(105, 2))
                #Fill first column
                for i in range(0, range_size):
                        euclidean_distance[i][0] = (target_data[0] - final_zero[i][0]) ** 2 + (target_data[1] - final_zero[i][1]) ** 2 + (target_data[2] - final_zero[i][2]) ** 2 + (target_data[3] - final_zero[i][3]) ** 2  
                #Fill second column
                for i in range(0, range_size):
                        euclidean_distance[i][1] = final_zero[i][4]
     
                return euclidean_distance

        def find_min(self, euclidean_distance_array, target_number, range_size):
                min = 1000
                for i in range(0, range_size):
                        if(euclidean_distance_array[i][1] == target_number):
                                if(min > euclidean_distance_array[i][0]):
                                        min = euclidean_distance_array[i][0]
                return min

        def find_max(self, euclidean_distance_array, target_number, range_size):
                max = -1000
                for i in range(0, range_size):
                        if(euclidean_distance_array[i][1] == target_number):
                                if(max < euclidean_distance_array[i][0]):
                                        max = euclidean_distance_array[i][0]

                return max
        
        def zero(self, the_set, range_size):
                column_one_mean = self.calculate_mean(the_set, 0)
                column_two_mean = self.calculate_mean(the_set, 1)
                column_three_mean = self.calculate_mean(the_set, 2)
                column_four_mean = self.calculate_mean(the_set, 3)
                column_one_std = self.calculate_std(the_set, 0)
                column_two_std = self.calculate_std(the_set, 1)
                column_three_std = self.calculate_std(the_set, 2)
                column_four_std = self.calculate_std(the_set, 3)

        
                zeroed = np.zeros(shape=(range_size,4))
                
                ## Create the zero array
                self.z_filler(the_set, 0, zeroed, column_one_mean, column_one_std, range_size)        
                self.z_filler(the_set, 1, zeroed, column_two_mean, column_two_std, range_size)        
                self.z_filler(the_set, 2, zeroed, column_three_mean, column_three_std, range_size)        
                self.z_filler(the_set, 3, zeroed, column_four_mean, column_four_std, range_size)        
                
                targets = np.zeros(shape=(range_size, 1))
                for i in range(0, range_size):
                        targets[i][0] = the_set[i][4]

                final_zero = np.column_stack((zeroed, targets))
                
                return final_zero

        def return_ranges(self, final_zero, target_data, range_size):
                euclidean_distance_array = self.euclid(final_zero, target_data, range_size)
                zero_min = self.find_min(euclidean_distance_array, 0, range_size)
                zero_max = self.find_max(euclidean_distance_array, 0, range_size)
                one_min = self.find_min(euclidean_distance_array, 1, range_size)
                one_max = self.find_max(euclidean_distance_array, 1, range_size)
                two_min = self.find_min(euclidean_distance_array, 2, range_size)
                two_max = self.find_max(euclidean_distance_array, 2, range_size)
                ranges = [zero_min, zero_max, one_min, one_max, two_min, two_max]
                return ranges
        
        def guess(self, ranges, euclidean_distance_array, range_size, row_number):
                average_zero = ((ranges[0] + ranges[1]) / 2)
                average_one = ((ranges[2] + ranges[3]) / 2)
                average_two = ((ranges[4] + ranges[5]) / 2)
                distance_zero = (average_zero - euclidean_distance_array[row_number][0]) ** 2
                distance_one = (average_one - euclidean_distance_array[row_number][0]) ** 2
                distance_two = (average_two - euclidean_distance_array[row_number][0]) ** 2
                answer = 0

                if distance_zero < distance_one and distance_zero < distance_two:
                        answer = 0
                elif distance_one < distance_zero and distance_one < distance_two:
                        answer = 1
                else:
                        answer = 2
                
                return answer

        def train(self, training_set, target_data, range_size):
                final_zero = self.zero(training_set, range_size)
                ranges = self.return_ranges(final_zero, target_data, range_size)
                return ranges
        
        def predict(self, test_set, target_data, range_size, the_ranges):
                final_zero = self.zero(test_set, range_size)
                euclidean_distance_array = self.euclid(final_zero, target_data, range_size)
                total_right = 0
                len_euclid = len(euclidean_distance_array)
                guess_array = np.zeros(shape=(len_euclid,1))
                for i in range(0, len_euclid):
                        guess_array[i] = self.guess(the_ranges, euclidean_distance_array, range_size, i)

                for i in range(0, range_size):
                        if (test_set[i][4] == guess_array[i]):
                                total_right += 1
                return float(total_right) / float(range_size)



hard_coded_classifier = HardCodedClassifier()
target_data = test_set[26]
the_ranges = hard_coded_classifier.train(training_set, target_data, 105)
total_right = hard_coded_classifier.predict(test_set, target_data, 35, the_ranges)
accuracy = total_right * 100
print(accuracy)
print ("% accuracy rate for Iris.csv")

# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)

	accuracy = getAccuracy(testSet, predictions)
	print("Accuracy: "),
        print(accuracy)
        print("%")
main()
