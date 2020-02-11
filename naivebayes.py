import sys
import io
import math
class NaiveBayes(object):

    def __init__(self, trainingData, trainingLabel):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        train = trainingData.splitlines()
        label = trainingLabel.splitlines()
        self.neutral_dict = {}
        self.positive_dict = {}
        self.negative_dict = {}
        label_list = []
        for l in label:
            label_list.append(l)
        for idx, sentence in enumerate(train):
            token_list = sentence.split(' ')
            label = label_list[idx]
            for token in token_list:
                if label == 'neutral':
                    if token not in self.neutral_dict:
                        self.neutral_dict[token] = 1
                    else:
                        self.neutral_dict[token] += 1
                elif label == 'positive':
                    if token not in self.positive_dict:
                        self.positive_dict[token] = 1
                    else:
                        self.positive_dict[token] += 1
                elif label == 'negative':
                    if token not in self.negative_dict:
                        self.negative_dict[token] = 1
                    else:
                        self.negative_dict[token] += 1
        
        neutral_sum = 0
        for v in self.neutral_dict.values():
            neutral_sum += v
        pos_sum = 0
        for v in self.positive_dict.values():
            pos_sum += v
        neg_sum = 0
        for v in self.negative_dict.values():
            neg_sum += v

        neutral_distinct = len(self.neutral_dict)
        pos_distinct = len(self.positive_dict)
        neg_distinct = len(self.negative_dict)

        for v in self.neutral_dict:
            self.neutral_dict[v] = (self.neutral_dict[v]) / (neutral_sum)
        for v in self.positive_dict:
            self.positive_dict[v] = (self.positive_dict[v]) / (pos_sum)
        for v in self.negative_dict:
            self.negative_dict[v] = (self.negative_dict[v]) / (neg_sum)
        
        sum_all = neutral_sum + pos_sum + neg_sum

        self.p_neu = neutral_sum / sum_all
        self.p_pos = pos_sum / sum_all
        self.p_neg = neg_sum / sum_all

        self.neu_denominator = neutral_sum + neutral_distinct
        self.pos_denominator = pos_sum + pos_distinct
        self.neg_denominator = neg_sum + neg_distinct

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        token_list = sentence.split(' ')
        neutral = math.log(self.p_neu)
        pos = math.log(self.p_pos)
        neg = math.log(self.p_neg)

        for token in token_list:
            if token in self.neutral_dict:
                neutral += math.log(self.neutral_dict[token])
            else:
                neutral += math.log(1 / self.neu_denominator)

            if token in self.positive_dict:
                pos += math.log(self.positive_dict[token])
            else:
                pos += math.log(1 / self.pos_denominator)

            if token in self.negative_dict:
                neg += math.log(self.negative_dict[token])
            else:
                neg += math.log(1 / self.neg_denominator)
        return {'positive': pos, 'negative': neg, 'neutral': neutral}

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """

        test = testData.splitlines()
        
        result = []
        for sentence in test:
            p = self.estimateLogProbability(sentence)
            label = max(p, key=p.get)
            result.append(label)
        return result

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 5:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    train_label = sys.argv[2]
    test_txt = sys.argv[3]
    test_label = sys.argv[4]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()
    
    with io.open(train_label, 'r', encoding='utf8') as f:
        train_label = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = NaiveBayes(train_data, train_label)
    evaluation = model.testModel(test_data)

    with io.open(test_label, 'w', encoding='utf8') as f:
        for label in evaluation:
            f.write(label + '\n')


