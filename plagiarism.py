import pandas as pd
import numpy as np
import re
import json
import os

PATH = os.path.dirname(os.path.realpath(__file__))


def cleanText(Text):
    # Remove new lines within message
    cleanedText = Text.replace('\n', ' ').lower()
    # Deal with some weird tokens
    cleanedText = cleanedText.replace("\xc2\xa0", "")
    # Remove punctuation
    cleanedText = re.sub('([,])', '', cleanedText)
    # Remove multiple spaces in message
    cleanedText = re.sub(' +', ' ', cleanedText)
    cleanedText = cleanedText.encode('ascii', 'ignore').decode('ascii')
    return cleanedText


def getData():
    df = pd.read_csv(PATH + '\\plagcheckfile.csv')
    listTexts = df['Text'].values.tolist()
    finallist = []
    print(listTexts)
    for i in range(0, 4):
        textDictionary = {"tag": i}
        listTexts_i = cleanText(listTexts[i])
        print(listTexts_i)
        textDictionary.update(texts=listTexts_i)
        finallist.append(textDictionary)
    finalDictionary = {"intents": finallist}
    return finalDictionary


combinedDictionary = dict()
print('Getting Data ... ')
combinedDictionary.update(getData())
print('The length of the dictionary is : ', len(combinedDictionary))

print('Saving content ... ')
np.save(PATH + '\\textDictionary.npy', combinedDictionary)

with open(PATH + '\\file.txt', 'w') as file:
    file.write(json.dumps(combinedDictionary))
