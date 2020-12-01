def find_sentence(start, end, text, length_of_text):
    """
    Finds first and last index of a sentences. Including the dot
    :param start: First char index of comment
    :param end:  Last char index of comment
    :param text: Parsed text from Word doc
    :param length_of_text: Length of of pre-parsed text
    :return: Tuple of integers, start and end respectively.
    """
    end_tmp = end
    start_tmp = start

    # Finds the start of the sentence (+ 1 from prev dot or 0 if this is the first sentences).
    start_index = start
    while text[start_tmp] != '.' and text[start_tmp] != '\r':
        if start_tmp == 0:
            break
        start_tmp = start_tmp - 1
        start_index = start_tmp

    # Finds end of the sentence (- 1 from prev dot or end of the text if there is no dot in the last sentence).
    end_index = end
    while text[end_tmp] != '.' and text[end_tmp] != '\r':
        if end_tmp == length_of_text:
            end_index = length_of_text
            break
        end_tmp = end_tmp + 1
        if end_tmp >= len(text):
            break
        end_index = end_tmp

    # TODO: Clean and check indexes
    # Correction of the indexes
    if text[start_index] == '.':
        start_index = start_index + 2
    elif text[start_tmp] == '\r':
        start_index = start_index + 1

    if text[end_index] == '.':
        end_index = end_index + 1

    return start_index, end_index
