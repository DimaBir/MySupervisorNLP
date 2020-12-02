import os
import pandas as pd
import matplotlib.pyplot as plt

from nltk import word_tokenize, pos_tag

from models import word2vec
from text_utils import parse_files
from utils import find_doc_files_path

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    data_folder = r"data"

    full_path = os.path.join(root, data_folder)

    # TODO: Make loop
    file_paths = find_doc_files_path(root=full_path)

    # Proceed file by file
    filename = r"D:\MySupervisorNLP\wordy_ds.csv"
    # parse_files(file_paths, filename, comment_to_filter="wordy", only_with_comment=True)

    # TODO: Add this as separate functions
    # Read CSV to panda
    df = pd.read_csv(filename, encoding='latin_1')
    wordy_sentences = df.loc[df['Class'] == 1]

    word2vec(wordy_sentences)
    # clear_sentences = df.loc[df['Class'] == 0].sample(n=len(wordy_sentences))
    #
    # tags = []
    # for sentence in wordy_sentences['Sentence']:
    #     text = word_tokenize(sentence)
    #     for tuple in pos_tag(text):
    #         tags.append(tuple[1])
    #
    # print(tags)
    #
    # dictionary = {}
    # for current_tag in tags:
    #     dictionary[current_tag] = len([tag for tag in tags if current_tag == tag])
    #
    # print(dictionary)
    #
    # keys = []
    # count = []
    # for key in sorted(dictionary):
    #     keys.append(key)
    #     count.append(dictionary[key])
    #
    # plt.bar(keys, count, align='center', color=['red'])
    # plt.xticks(range(len(keys)), keys, rotation='vertical')
    # plt.ylabel('Count')
    # plt.legend('W')
    #
    # plt.show()

    # tags_df = pd.DataFrame(tags, columns=["Tags"])
    # print(tags_df)





