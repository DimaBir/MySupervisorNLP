import nltk
import win32com.client as win32

from sentence_utils import find_sentence


def split_text_to_sentences(text):
    """
    Splits text to sentences. Pays special attention to end-of-the sentence chars.
    :param text: Parsed word text
    :return: list of sentences
    """
    text = text.replace("\r", ". ")
    text = text.replace(" .", ".")

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences


def get_comments_and_sentences(filepath):
    """
    Parses word document. Extracts sentences and bind comments to them. Extracts and splits text to sentences.
    :param filepath: Absolute path to word file.
    :return: A dictionary with sentence and comment, split sentences list.
    """
    result_dict = {}
    word = win32.gencache.EnsureDispatch('Word.Application')
    word.Visible = False
    doc = word.Documents.Open(filepath)
    doc.Activate()
    activeDoc = word.ActiveDocument
    all_sentences = split_text_to_sentences(activeDoc.Range().Text)
    delta = -1
    for c in activeDoc.Comments:
        if c.Ancestor is None:  # checking if this is a top-level comment
            delta = delta + 1
            # if "wordy" in c.Range.Text.lower():
            print("Comment by: " + c.Author)
            print("Comment text: " + c.Range.Text)  # text of the comment
            print("Regarding: " + c.Scope.Text)  # text of the original document where the comment is anchored
            sentence_range = find_sentence(c.Scope.Start - delta, c.Scope.End - delta, activeDoc.Range().Text,
                                           activeDoc.Range().End)

            sentence = activeDoc.Range().Text[sentence_range[0]:sentence_range[1]]
            comment = c.Range.Text
            regarding = c.Scope.Text
            result_dict[sentence] = (comment, regarding)
            print(f"Sentence: \n{activeDoc.Range().Text[sentence_range[0]:sentence_range[1]]}")
            if len(c.Replies) > 0:  # if the comment has replies
                print("Number of replies: " + str(len(c.Replies)))
                for r in range(1, len(c.Replies) + 1):
                    print("Reply by: " + c.Replies(r).Author)
                    print("Reply text: " + c.Replies(r).Range.Text)  # text of the reply
    doc.Close()
    word.Quit()
    return result_dict, all_sentences
