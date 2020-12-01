import os

from text_utils import get_comments_and_sentences
from utils import find_doc_files_path

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    data_folder = r"data"

    full_path = os.path.join(root, data_folder)

    # TODO: Make loop
    filename = r"C:\Users\dmitry.v\PycharmProjects\MySupervisorUtils\dataset2.csv"
    file_paths = find_doc_files_path(root=full_path)

    # Proceed file by file
    for filepath in file_paths:
        try:
            sent_comm_dict, all_sentences = get_comments_and_sentences(filepath=filepath)
            # print(sent_comm_dict)
        except Exception as e:
            print(e)
        finally:
            continue

