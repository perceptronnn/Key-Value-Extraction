import pandas as pd 
from nltk.stem.wordnet import WordNetLemmatizer
import sys
import random 
import nltk
import os
list_path = './list.txt'
with open(list_path, 'r') as f:
    files = f.readlines()
dir_path = './raw_data/' #path to directory where inputs files are stored
out_path = './cleaned/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
pat = 'DEBUG:::'
for file in files:
    try:
        path = './' + file[:-1]
        #print(path)#file path
        curr_dir = out_path + file[:-5]
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        out_dir = curr_dir + '/lemmatized/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        my_file = open(dir_path + file[:-1] , 'r') #open file
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
            
        train_df = pd.read_table(train_out, sep = '\t', header = None)
        test_df = pd.read_table(test_out, sep = '\t', header = None)
        train_df = train_df.iloc[1:] #to remove 'a\ta\ta'
        test_df = test_df.iloc[1:] #to remove 'a\ta\ta'
        #pat = 'DEBUG:::' #apply filter to remove pattern
        train_filter = train_df[train_df.columns[0]].str.contains(pat, na = False)
        test_filter = test_df[test_df.columns[0]].str.contains(pat, na = False)
        train_df = train_df[~train_filter]
        test_df = test_df[~test_filter] 
        train_df = train_df.drop(train_df.columns[[-1,]], axis=1) #drop the last column
        test_df = test_df.drop(test_df.columns[[-1,]], axis=1)
        
        lmtzr = WordNetLemmatizer()
        for index, row in train_df.iterrows():
            if type(row[0]) is str:
                word = lmtzr.lemmatize(row[0].lower())
                row[0] = str(word)

        for index, row in test_df.iterrows():
            if type(row[0]) is str:
                word = lmtzr.lemmatize(row[0].lower())
                row[0] = str(word)
        
        if (train_df.shape[0] > 30,000):
            stepsize = 30000
            for id, i in enumerate(range(0,train_df.size,stepsize)): 
                start = i 
                end = i + stepsize-1 #neglect last row ...
                train_df.ix[start:end].to_csv(out_dir + 'train'+str(id)+'.tsv', sep='\t', index = False, header = None)
        else:
            train_df.to_csv(out_dir+'train.tsv', sep='\t', index = False, header = None)
        test_df.to_csv(out_dir +'test.tsv', sep='\t', index = False, header = None)
        
        '''
        df = pd.read_table(dir_path + file[:-1], sep = '\t', header = None)
        filter = df[df.columns[0]].str.contains(pat, na=False)
        df = df[~filter]
        df = df.drop(df.columns[[-1,]], axis=1)
        row_count = df.shape[0]
        split_point = int(row_count*7/10)
        test_data, train_data = df[split_point:], df[:split_point]
        train_data = train_data[1:]
        if not os.path.exists(out_path + file[:-5]):
            os.makedirs(out_path + file[:-5])
        train_data.to_csv(out_path + file[:-5]+'/train_'+file, sep='\t', index = False, header = None)
        test_data.to_csv(out_path + file[:-5]+'/test_'+file, sep='\t', index = False, header = None)
        '''
    except Exception as e: print(file +'	' +str(e))
    #except:
        #print(file)