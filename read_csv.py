######################################################################################
# Matthew Hansen - read_csv.py reads the iris.csv file into a numpy array, and then
#                  goes about creating a training and test set. 

######################################################################################
# This is where I import information
import numpy as np
import math

######################################################################################
# This is the section where I load the dataset
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

######################################################################################
# This section is where I randomize the entire set
for i in range(0, 50):           #50 was chosen randomly, but it would be 
    np.random.shuffle(combined)  #nice to change this to n-tier folding

######################################################################################
# This section is where I split the set into the training set 70% and the test set 30%
len_combined = len(combined)
seventy_percent = int(math.floor(len_combined * .7))
thirty_percent = len_combined - seventy_percent

training_set = np.zeros(shape=(seventy_percent, 5))
test_set = np.zeros(shape=(thirty_percent, 5))

for i in range(0, seventy_percent):
    training_set[i] = combined[i]

for i in range(0, thirty_percent):
    test_set[i] = combined[i + seventy_percent]



###################################################################################
# This section is where I create the HardCoded Classifier class
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
####################################################################################
# This section is where I instantiate the HardCoded Classifier and train and test it

hard_coded_classifier = HardCodedClassifier()
hard_coded_classifier.train(training_set)
hard_coded_classifier.predict(test_set)

####################################################################################                
# Determine the accuracy of your classifier's predictions and report the result as a percentage
number_right = hard_coded_classifier.accuracy(test_set)
accuracy = (float(number_right) / float(thirty_percent)) * 100
print accuracy,
print ("% accuracy rate")

###################################################################################
# Create a new public repository at GitHub and push your code to it.
print("https://github.com/diabloazul14/Experiment_Shell")
