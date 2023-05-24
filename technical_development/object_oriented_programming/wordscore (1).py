def score_word(word, wild_tile1=None, wild_tile2=None):
    """
    Scores the scrabble word

    This function simply produces a score
    for the scrabble word using a scores dictionary.
    Wild card tiles are scored as 0 points.

    Parameters
    ----------
    word : int
        A string of character tiles.
    wild_tile1 : int
        The position of a wild tile
    wild_tile2 : int
        The position of a wild tile

    Returns
    -------
    int
        The word score
    """
    scores_dict = {"A": 1, "C": 3, "B": 3, "E": 1, "D": 2, "G": 2,
         "F": 4, "I": 1, "H": 4, "K": 5, "J": 8, "M": 3,
         "L": 1, "O": 1, "N": 1, "Q": 10, "P": 3, "S": 1,
         "R": 1, "U": 1, "T": 1, "W": 4, "V": 4, "Y": 4,
         "X": 8, "Z": 10}
    
    score = 0
    for i in range(len(word)):
        # Wild cards are scored as 0 points. 
        # A word that just consists of two wildcards can be made, should be outputted and scored as 0 points.
        if i == wild_tile1 or i == wild_tile2:
            continue
        score += scores_dict[word[i]]

    return score