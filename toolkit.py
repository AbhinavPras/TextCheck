import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import json
import os

stemmer = LancasterStemmer()

PATH = os.path.dirname(os.path.realpath(__file__))

with open(PATH + "\\file.txt") as file:
    dataset = json.load(file)

list_words = []
labels = []
docs_x = []  # Lquestions.
docs_y = []  # texts to look out for

for intent in dataset["intents"]:
    for pattern in intent["texts"]:
        split_words = nltk.word_tokenize(pattern)
        list_words.extend(split_words)
        docs_x.append(split_words)  # Adding the words to docs_x.
        # For each pattern, it says what Tag it is a part of.
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])  # Adds the tags in the labels list.

list_words = [stemmer.stem(w.lower())
              for w in list_words if w != "?"]  # Converts to lower case
# Removes duplicate words => converted back into a sorted list
list_words = sorted(list(set(list_words)))

labels = sorted(labels)

training = []  # contains the words.
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    split_words = [stemmer.stem(w.lower()) for w in doc]

    for w in list_words:
        if w in split_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)


# bag_of_words function will transform our string input to a bag of words using our created words list
def bag_of_words(s, list_words):
    bag = [0 for _ in range(len(list_words))]

    inp_str_words = nltk.word_tokenize(s)
    inp_str_words = [stemmer.stem(word.lower()) for word in inp_str_words]

    for search_element in inp_str_words:
        for i, w in enumerate(list_words):
            if w == search_element:
                bag[i] = 1

    return numpy.array(bag)


training = numpy.array(training)
output = numpy.array(output)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)  # Model gets trained.

try:
    model.predict(PATH + "\\model.tflearn")
    model.load()
except:
    # Fitting our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training.
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    # we can save it to the file model.tflearn for use in other scripts.
    model.save(PATH + "\\model.tflearn")


def check():

    print("Type an answer (Enter 'quit' to stop)")
    while True:
        inp = input("You: ")
        inp = inp.lower()
        if inp == "quit":
            break

        # we predict the most matching text from the training dataset that matches with the input text.
        results = model.predict([bag_of_words(inp, list_words)])
        # we get thr index of the result.
        results_index = numpy.argmax(results)
        # we get the tag of the most matching text with the input text.
        tag = labels[results_index]

        for tg in dataset["intents"]:
            if tg['tag'] == tag:
                response = tg['texts']
        print("Most similar text to the input: ")
        print(response)  # we show the most matching text with the input text.
        # we divide the response text (the most matching text with the input text)
        split_sent_resp = nltk.sent_tokenize(response)
        # we put the sentenses in a different list for future work.
        list_sents = split_sent_resp

        # we split the input text into sentenses
        split_sent_inp = nltk.sent_tokenize(inp)
        len_inp = len(split_sent_inp)  # number of sentenses in the input text.
        list_sents.extend(split_sent_inp)
        # we get the total number of sentenses in input and response text, including the duplicate sentenses.
        total_len = len(list_sents)
        # we are making the list a set, so that duplicate sentenses get deleted.
        set_sents = set(list_sents)
        set_len = len(set_sents)
        # this gives us the number of duplicate sentenses, i.e, the copied or plaigarized sentenses.
        len_difference = total_len-set_len
        plag_percent = (1-((len_inp-len_difference)/len_inp))*100
        print("The percent of plagarism in the document is: ", plag_percent)


check()
