import time
from operator import itemgetter
from wordscore import score_word
import itertools
from string import ascii_uppercase as alphabet_letters


# For partial credit, your program should take less than one minute to run with 2 wildcards in the input. 
# For full credit, the program needs to run with 2 wildcards in less than 30 seconds

def get_valid_word_combinations(word, all_valid_eng_words):
    """
    Given a list of character tiles, returns all english word 
    combinations that can be created from those letters.

    This function creates all potential charcter combinations,
    creates all permutations of the combinations
    and checks the validity of the words with the english list
    of words. Additionally, this function creates wild card
    permutations with the english alphabet. Each valid word
    is scored with a scoring function.

    Parameters
    ----------
    word : str
        A string of character tiles.
    all_valid_eng_words : list
        A list of all english words

    Returns
    -------
    list
        A list of all valid word combinations in tuple format (score, word)
    """
    
    # Step 1: create all potential combinations
    combinations = []
    permutations = set([])
    for i in range(1, len(word)):
        combinations.extend(list(itertools.combinations(word, i+1)))
    # Step 2: create all permutations
    for combo in combinations:
        permutations.update(list(itertools.permutations(combo)))        
    
    # Step 3: check all words
    word_combinations = []
    checked_words = set([])
    for potential_word in permutations:
        potential_word = ''.join(potential_word)
        
        # skip any words we already checked
        if potential_word in checked_words:
            continue
            
        # First, check for no wild cards so that the higher scoring word gets added first
        # if there is no wild ard, check the validity of the word
        if '*' not in potential_word and '?' not in potential_word: 
            if potential_word in all_valid_eng_words:
                word_combinations.append((score_word(potential_word), potential_word))
            
        # first, check for wild cards
        elif '*' in potential_word:
            # replace wild card with each letter of the alphabet
            for letter in alphabet_letters:
                wild_word = potential_word.replace('*', letter)
                # check for a second wild card
                if '?' in wild_word:
                    for second_letter in alphabet_letters:
                        wild_wild_word = wild_word.replace('?', second_letter)
                        if wild_wild_word in all_valid_eng_words:
                            word_combinations.append((score_word(wild_wild_word, potential_word.find('*'), potential_word.find('?')), wild_wild_word))
                else:
                    if wild_word in all_valid_eng_words:
                        word_combinations.append((score_word(wild_word, potential_word.find('*')), wild_word))

                        
        # check of ? only wild card
        elif '?' in potential_word:
            for letter in alphabet_letters:
                wild_word = potential_word.replace('?', letter)
                if wild_word in all_valid_eng_words:
                        word_combinations.append((score_word(wild_word, potential_word.find('?')), wild_word))

        # Step 3: update list of all checked words with our permutations
        checked_words.update(potential_word)
        
    return word_combinations



def run_scrabble(word):
    """
    Produces a list of valid scrabble words with scores and the
    number of words that could be created by the characters
    provided.

    This function checks the input for errors, reads the
    english words in, calls the function to get word combinations,
    removes duplicates, sorts the word list, and returns the count
    of words.

    Parameters
    ----------
    word : str
        A string of character word tiles

    Returns
    -------
    list
        List with two elements: the list of tuples in format (score, word), and the count of valid words
    """
    start_time = time.time()
    
    word = word.upper()
    # Step 0: Error Checking
    # Allow anywhere from 2-7 character tiles (letters A-Z, upper or lower case) to be inputted
    if len(word) > 7 or len(word) < 2:
        return "Scrabble word is either too long or too short. Please enter a word between 2 and 7 character tiles."
    if not all(character.isalpha() or character == '?' or character == '*' for character in word):
        return "Scrabble word contains a character that is not allowed. Please limit character tiles toletters A-Z, * and ?"
    if word.count('*') > 1 or word.count('?') > 1:
        return "Too many wild card tiles. A maximum of one * and one ? are allowed."
    
    # Step up: Read in the valid english words
    with open("sowpods.txt","r") as infile:
        raw_input = infile.readlines()
        all_valid_eng_words = [datum.strip('\n') for datum in raw_input]
        
    # Step 1: Get all valid word combinations
    word_combinations = get_valid_word_combinations(word, all_valid_eng_words)
    
    # In a wildcard case where the same word can be made with or without the wildcard, display the highest score
    for idx, tuple1 in enumerate(word_combinations):
        for jdx, tuple2 in enumerate(word_combinations):
            if tuple1[1] == tuple2[1] and idx != jdx:
                if tuple1[0] > tuple2[0]:
                    word_combinations.remove(tuple2)
                else:
                    word_combinations.remove(tuple1)
            
    # Step 2: Sort words by scores
    sorted_word_list = sorted(word_combinations, key=lambda element: (element[0], element[1]))
    
    end_time = time.time()
    print("Time to run:", round(end_time - start_time, 2), "seconds")
    return sorted_word_list, len(sorted_word_list)