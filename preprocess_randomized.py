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

if (train_df.shape[0] > 30,000):
    stepsize = 30000
    for id, i in enumerate(range(0,train_df.size,stepsize)): 
        start = i 
        end = i + stepsize-1 #neglect last row ...
        train_df.ix[start:end].to_csv('train'+str(id)+'.tsv', sep='\t', index = False, header = None)
else:
    train_df.to_csv('train.tsv', sep='\t', index = False, header = None)
test_df.to_csv('test.tsv', sep='\t', index = False, header = None)