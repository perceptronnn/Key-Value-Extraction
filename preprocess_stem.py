#Script for complete preprocessing of a dataset from a particular product category from the dataset

#Imports 
import pandas as pd 
from nltk.stem.wordnet import WordNetLemmatizer
import sys
import random 
import nltk

#randomization
path = sys.argv[1] #file path
my_file = open(path, 'r') #open file
my_str = my_file.read() #read the whole file as a string
my_list = my_str.split('\n\n') #split the string using '\n\n' to get a list
random.Random(4).shuffle(my_list) #shuffle the list with seed 4
test_list = my_list[:len(my_list)/5]
train_list = my_list[len(my_list)/5:]
head = 'a\ta\ta\n\n'
new_train_str = '\n\n'.join(str(e) for e in train_list) #join the shuffled list to get a string
new_test_str = '\n\n'.join(str(e) for e in test_list)
new_train_str = head + new_train_str
new_test_str = head + new_test_str
train_out = './train_out.tsv'
test_out = './test_out.tsv'
with open(train_out, 'w') as f:
	f.write(new_train_str)
with open(test_out, 'w') as f:
	f.write(new_test_str)	

#cleaning 
train_df = pd.read_table(train_out, sep = '\t', header = None)
test_df = pd.read_table(test_out, sep = '\t', header = None)
train_df = train_df.iloc[1:] #to remove 'a\ta\ta'
test_df = test_df.iloc[1:] #to remove 'a\ta\ta'
pat = 'DEBUG:::' #apply filter to remove pattern
train_filter = train_df[train_df.columns[0]].str.contains(pat, na = False)
test_filter = test_df[test_df.columns[0]].str.contains(pat, na = False)
train_df = train_df[~train_filter]
test_df = test_df[~test_filter] 
train_df = train_df.drop(train_df.columns[[-1,]], axis=1) #drop the last column
test_df = test_df.drop(test_df.columns[[-1,]], axis=1)

#stemming
ps = nltk.stem.PorterStemmer()
for index, row in train_df.iterrows():
	if type(row[0]) is str:
		word = ps.stem(row[0])
    	row[0] = str(word)
for index, row in test_df.iterrows():
	if type(row[0]) is str:
		word = ps.stem(row[0])
    	row[0] = str(word)

if (train_df.shape[0] > 30000):
    trainFileList = ''
    stepsize = 30000
    for id, i in enumerate(range(0,train_df.size,stepsize)): 
        if i <= train_df.shape[0]:
            #print(i)
            start = i 
            end = i + stepsize-1 #neglect last row ...
            train_df.ix[start:end].to_csv('./model/train'+str(id)+'.tsv', sep='\t', index = False, header = None)
            trainFileList = trainFileList + 'train' + str(id) + '.tsv,'
    trainFileList = 'trainFileList = ' + trainFileList[:-1] + '\n'
    #print(trainFileList)
else:
    train_df.to_csv('./model/train.tsv', sep='\t', index = False, header = None)
    trainFileList = 'trainFile = train.tsv\n'
test_df.to_csv('./model/test.tsv', sep='\t', index = False, header = None)

prop = ['#location where you would like to save (serialize to) your\n', '#classifier; adding .gz at the end automatically gzips the file,\n', '#making it faster and smaller\n', 'serializeTo = ner-model.ser.gz\n', '\n', '#structure of your training file; this tells the classifier\n', '#that the word is in column 0 and the correct answer is in\n', '#column 1\n', 'map = word=0,answer=1\n', '\n', "#these are the features we'd like to train with\n", '#some are discussed below, the rest can be\n', '#understood by looking at NERFeatureFactory\n', 'useClassFeature=true\n', 'useWord=true\n', 'useNGrams=true\n', '#no ngrams will be included that do not contain either the\n', '#beginning or end of the word\n', 'noMidNGrams=true\n', 'useDisjunctive=true\n', 'maxNGramLeng=6\n', 'usePrev=true\n', 'useNext=true\n', 'useSequences=true\n', 'usePrevSequences=true\n', 'maxLeft=1\n', '#the next 4 deal with word shape features\n', 'useTypeSeqs=true\n', 'useTypeSeqs2=true\n', 'useTypeySequences=true\n', 'wordShape=chris2useLC\n']
prop.insert(0, trainFileList)
prop.insert(0, '# location of the training file\n')
myfile = open('./model/austen.prop', 'w')
for item in prop:
    myfile.write(item)
myfile.close()