import os

from text_utils import parse_files
from utils import find_doc_files_path

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    data_folder = r"data"

    full_path = os.path.join(root, data_folder)

    # TODO: Make loop
    file_paths = find_doc_files_path(root=full_path)

    # Proceed file by file
    filename = r"D:\MySupervisorNLP\dataset2.csv"
    parse_files(file_paths, filename)





