# Instructions for training and testing a classifier for a particular product category
1. Clone/Download this repository and make it the present working directory. <br/>
2. For preprocessing, run any of the preprocessing script among *preprocess_lemmatize.py*, *preprocess_stem.py* and *preprocess_randomize.py* along with the path to the data file for the intended category. For example: <br />
  ```
  pyhton preprocess_stemmed.py /home/anurag/raw_data/lungi_3_45.tsv
  ```
3. Change the present working directory to the *model* directory and run the following command:<br/>
  ```
  java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop austen.prop
  ```
  This will train a model named *ner-model.ser.gz*<br />
  4. Test the model using the following command: <br />
  ```
  java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -testFile test.tsv
  ```
