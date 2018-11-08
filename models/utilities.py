def group_phrases(keywords, word_list):
    """

    :param word_list: List of words (the raw text being used.)
    :return: List of words with phrase keywords grouped together using "_".
    """
    for keyword in keywords:
        if "_" in keyword:
            keyword_split = keyword.split("_")
            first_word = keyword_split[0]
            num_words = len(keyword_split)
            index = 0
            while index < len(word_list):
                if index != 0:
                    index = index + 1

                try:
                    found_index = word_list[index:].index(first_word)
                    index = index + found_index
                    new_string = word_list[index]
                    found = True
                    if index + num_words < len(word_list):
                        for num in range(0, num_words):
                            if keyword_split[num] != word_list[index + num]:
                                found = False
                                break
                            elif num != 0:
                                new_string += "_" + word_list[index + num]
                        if found is True:
                            word_list[index] = new_string
                            for other_indices in range(index + 1, index + num_words):
                                word_list.pop(other_indices)
                except ValueError as e:
                    index = len(word_list) + 1

    return word_list


def convert_phrases(keywords, word2int_dict):
    """

    :param word2int_dict: Word2Int dict that has the integer conversions for the vocab.
    :return: word2int dict with the phrase keywords added to the end.
    """
    for keyword in keywords:
        if "_" in keyword:
            word2int_dict[keyword] = len(word2int_dict)
    return word2int_dict


def adjust_keywords(keywords):
    """

    :param keywords: List of keywords being used.
    :return: Keywords dict with phrases joined by "_".
    """
    new_keywords = {}
    for keyword, index in keywords.items():
        new_keyword = keyword
        if " " in keyword:
            new_keyword = keyword.replace(" ", "_")
        new_keywords[new_keyword] = keywords[keyword]
    return new_keywords