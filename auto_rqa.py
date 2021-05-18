#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
AUTO-RQA FOR DREAMS
@author: Katarina Laken
April 2021
'''

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import spacy
import neuralcoref
import gensim
import nltk
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm') #import spacy pipeline, make sure you have "en_core_web_sm" installed
neuralcoref.add_to_pipe(nlp, greedyness=0.52)

os.chdir('/home/hp/Documenten')
embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./WORD2VEC_pretrained/GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

SIMILARITY_THRESHOLD = 0.2

RQA_MEASURES = {} #dictionary that will be filled with the dream names (key) and a dictionary (value) containing the values for the keys 'determinism', 'laminarity', 'rec', and 'linemax'.

################### FROM RAW DREAM TO LIST OF EMBEDDINGS #######################

def preprocess_dream(raw_dream):
    #input: an _io.TextIOWrapper object containing a dream description
    #output: spacy.tokens.doc.Doc
    txt_as_string = raw_dream.read()
    txt_as_string = re.sub('-[A-Z]{3}-', '', txt_as_string)
    #title = re.search('QQQ([0-9])+\.', txt_as_string).group()
    #txt_as_string = re.sub(title, '', txt_as_string) #remove the title
    txt_as_string = re.sub(' (?=[^A-Za-z])', '', txt_as_string) #to remove weird spaces
    txt = nlp(txt_as_string)
    return(txt)

def get_pronoun_reference(pronoun):
    #input: a spacy token containing a pronoun
    #output: the main word the pronoun referred to (1 lemma)

    word = pronoun.text.lower()

    #for a first or second person pronoun, don't look for a lemma
    if re.search('^(i|me|my|mine|myself)$', word) is not None:
        return('I')
    elif re.search('^(you|your|yours|yourself|yourselves)$', word) is not None:
        return('you')
    elif re.search('^(we|us|our|ours|ourselves)$', word) is not None:
        return('we')

    #for all other pronouns, look for a coreference
    if pronoun._.in_coref:
        reference_span = pronoun._.coref_clusters[0].main #get the main mention of the first identified coref cluster. Type = space.tokens.span.Span
        main_lemma = reference_span.root.lemma_ #the main lemma of the span. This can be a pronoun
        if re.search('PRON', main_lemma) is not None:
            return(reference_span.root.text.lower())
        else:
            return(main_lemma)
    else: #if there is none, just return the (lower-cased) pronoun
        return(word)

def remove_stopwords(lemma_list):
    #input: a list of lemmas
    #output: that same list with the stopwords (according to NLTK stopwords removed)
    for lemma in lemma_list:
        if lemma in stopwords.words('english'):
            lemma_list.remove(lemma)
    return(lemma_list)

def create_lemma_list(txt):
    #input: a spacy Doc of the dream description
    #output: a vector where words are replaced with indexes. Pronouns have the index of the main word they refer to. Words with the same lemma have the same index.

    lemma_list = [] #a list of lemmas that represent the text (each word is replaced with its lemma of the lemma it refers to, and appended to the list)
    num_vector = [] #a list of the same length were each lemma corresponds to an integer index Integer is the index of the first occurrence of the lemma in the lem_vector list. (NB I do not us this in this script, but it can be useful for further exploration of the data/other analyses).
    n_lemmas = 0

    for word in txt:
        #first: check if the word is not a punctuation mark (these do not get indexes)
        if re.search('^\W$', word.text) is None:
            lemma = word.lemma_
            #I only look for a reference if there is a pronoun; because I use embeddings to compute similarity, I expect words that refer to similar things to have very close embeddings and thus to show up anyway
            if re.search('PRON', lemma) is not None:
                lemma = get_pronoun_reference(word)
            if lemma in lemma_list: #if the lemma already occurred, get its index for num_vector
                lemma_index = lemma_list.index(lemma)
                num_vector.append(lemma_index)
            else:
                lemma_index = n_lemmas
                num_vector.append(lemma_index)
                n_lemmas += 1
            lemma_list.append(lemma)

    lemma_list = remove_stopwords(lemma_list)

    return(lemma_list)

def get_embeddings(lemma_list):
    #input: a list of lemmas that represent the text (each word is replaced with its lemma of the lemma it refers to, and appended to the list)
    #output: a list containing the embeddings corresponding to the lemmas of lem_vector. If all lemmas have an embedding it is the same length as lemma_list, but lemmas without embedding are skipped over

    embeddings_list = []
    lemma_list = ['an' if x == 'a' else x for x in lemma_list] #for some reason the lemma of the indefinite article 'a(n)' is 'a' in spacy and 'an' in word2vec so we should fix that first

    for lemma in lemma_list:
        try:
            embedding = embedding_model.get_vector(lemma)
            embeddings_list.append(embedding)
        except KeyError: #if I cannot find an embedding I just drop the word
            pass

    return(embeddings_list)

################################## RECURRENCE PLOT #############################

def get_cosine_similarity(x, y):
    #input: two numpy 1d arrays of floats
    #output: the cosine similarity between the two (1 float)

    cosine_similarity = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

    return(cosine_similarity)

def get_cosine_sim_matrix(embedding_vector):
    #input: a vector of word embeddings
    #output: a matrix containing the cosine similarity between each two word embeddings

    n_embeddings = len(embedding_vector)
    embeddings = np.array(embedding_vector) #for a dream of 199 words, this array has the shape (199, 300)
    cosine_sim_matrix = np.empty((n_embeddings, n_embeddings))

    #for loop to fill the matrix
    for x_index, x_embedding in enumerate(embeddings):
        for y_index, y_embedding in enumerate(embeddings):
            cosine_similarity = get_cosine_similarity(x_embedding, y_embedding)
            cosine_sim_matrix[x_index, y_index] = cosine_similarity

    return(cosine_sim_matrix)

def create_recurrence_plot(cosine_sim_matrix, filename):
    #input: the cosine similarity matrix
    #output: nothing
    #this function creates a nice recurrence plot
    #make sure there is a map called 'plots' in your working directory!

    rc_plot = plt.matshow(cosine_sim_matrix)
    plot_name = re.search('QQQ([0-9]+).tok.txt', filename).group(1)
    plot_name = "rc_plot" + plot_name + ".jpg"
    plot_path = "/home/hp/Documenten/labrot I/plots/" + plot_name #for some unknown reason this only works with the absolute path. If the absolute path doesn't work, try the relative path from the working directory. If I don't specify a path, it does save the file in my working directory.
    plt.savefig(plot_path)

def get_recurrency(cosine_sim_matrix, n_words):
    #input: cosine similarity matrix
    #       n_words is integer indicating the length of the sides of the matrix
    #output: number of recurrent points
    #excluding main diagonal
    n_recurrence_points = 0
    for x_index in range(n_words):
        for y_index in range(n_words):
            if x_index != y_index: #if we are on the diagonal line, do not count
                cosine_similarity = cosine_sim_matrix[x_index, y_index]
                if cosine_similarity > SIMILARITY_THRESHOLD:
                    n_recurrence_points += 1
    return(n_recurrence_points)

def get_determinism(cosine_sim_matrix, n_words):
    #input: cosine similarity matrix
    #       n_words is integer indicating the length of the sides of the matrix
    #       n_recurrent: n of points that is recurrent
    #output: n points that is on a diagonal line
    n_on_diag = 0
    for x_index in range(n_words):
        for y_index in range(n_words):
            if x_index != y_index: #if we are on the main diagonal, do not count
                cosine_similarity = cosine_sim_matrix[x_index, y_index]
                if cosine_similarity > SIMILARITY_THRESHOLD:
                    #check if recurrence point is on a diagonal line. Minimum line length = 2
                    if x_index > 0 and y_index > 0: #can we do a step back?
                        one_back = cosine_sim_matrix[x_index-1, y_index-1]
                        if one_back > SIMILARITY_THRESHOLD:
                            n_on_diag += 1
                            continue #if the point is on a diagonal, go to the next point
                            #otherwise I would count points double
                    if x_index < n_words-1 and y_index < n_words-1: #can we take a step ahead?
                        one_ahead = cosine_sim_matrix[x_index+1, y_index+1]
                        if one_ahead > SIMILARITY_THRESHOLD:
                            n_on_diag += 1
    return(n_on_diag)

def get_laminarity(cosine_sim_matrix, n_words):
    #input: cosine sim matrix and number of words (integer)
    #output: number of points that are on a vertical line (integer)
    n_on_vert = 0
    for x_index in range(n_words):
        for y_index in range(n_words):
            #here I do want to count the diagonal
            cosine_similarity = cosine_sim_matrix[x_index, y_index]
            if cosine_similarity > SIMILARITY_THRESHOLD:
                #check if I can go one down
                if y_index > 0:
                    one_down = cosine_sim_matrix[x_index, y_index-1]
                    if one_down > SIMILARITY_THRESHOLD:
                        n_on_vert += 1
                        continue #if the point is on a vertical, go to the next point
                        #otherwise I would count points double
                #check if I can go one up
                if y_index < n_words-1:
                    one_up = cosine_sim_matrix[x_index, y_index+1]
                    if one_up > SIMILARITY_THRESHOLD:
                        n_on_vert += 1
    return(n_on_vert)

def get_linemax_OLD(cosine_sim_matrix, n_words):
    #input: the cosine similarity matrix
    #output: the length of the longest diagonal line
    longest_line = 0
    current_line_length = 0
    for x_index in range(n_words):
        for y_index in range(n_words):
            if x_index != y_index: #main diagonal should not be counted
                cosine_similarity = cosine_sim_matrix[x_index, y_index]
                if cosine_similarity > SIMILARITY_THRESHOLD:
                    current_line_length = 1 #begin een nieuwe lijn
                    on_diag_line = True
                    while on_diag_line:
                        x_next = x_index + current_line_length
                        y_next = y_index + current_line_length
                        if x_next < n_words and y_next < n_words: #check if not at a border
                            one_ahead = cosine_sim_matrix[x_next, y_next] #if not, get the cosine similarity of the next point
                            if one_ahead > SIMILARITY_THRESHOLD:
                                current_line_length += 1
                            else:
                                on_diag_line = False
                        else:
                            on_diag_line = False
            if current_line_length > longest_line:
                longest_line = current_line_length
    return(longest_line)

def get_line_lengths(cosine_sim_matrix, n_words):
    #input: the cosine sim matrix and the amount of words
    #output:a dictionary with all line lengths as key and the amount of lines of
    #       that length as value
    points_in_lines = []    #list with all the coordinates of recurrent points
                            #   already counted as lines
    line_lengths = {}
    for x_index in range(n_words):
        for y_index in range(n_words):
            coordinate = (x_index, y_index)
            if (x_index != y_index) and (coordinate not in points_in_lines): #not counting the main diagonal; also, check if we did not already count this point in some line
                cosine_similarity = cosine_sim_matrix[x_index, y_index]
                if cosine_similarity > SIMILARITY_THRESHOLD:
                    on_diag_line = True
                    line_length = 1 #start counting a line
                    while(on_diag_line):
                        x_next = x_index + line_length
                        y_next = y_index + line_length
                        if x_next < n_words and y_next < n_words: #check if not at a border
                            next_cosine_similarity = cosine_sim_matrix[x_next, y_next]
                            if next_cosine_similarity > SIMILARITY_THRESHOLD:
                                line_length += 1
                                continue

                        # so the rest of this block is only executed if we are not
                        # at a border and the next point is not a recurrent point

                        if line_length > 1: #only count lines, not just recurrent points
                            if line_length in line_lengths: #count line_length in dictionary
                                line_lengths[line_length] += 1
                            else:
                                line_lengths[line_length] = 1
                        on_diag_line = False
    return(line_lengths)

def get_line_length_entropy(line_lengths, n_words):
    #input:a dictionary with all line lengths as key and the amount of lines of
    #       that length as value
    #output: entropy
    length_probs = {} #dictionary storing the line lengths (key) with their relative probability (value)
    n_lines = 0
    if len(line_lengths) > 0:
        for line_length in line_lengths:
            n_lines += line_lengths[line_length]
        for line_length in line_lengths:
            length_probability = line_lengths[line_length] / n_lines
            length_probs[line_length] = length_probability

        #now, get the shannon's entropy for the probability distribution
        total_entropy = 0
        for prob in length_probs:
            prob_entropy = prob * np.log(prob)
            total_entropy += prob_entropy
        entropy = total_entropy/len(line_lengths)
        return(entropy)
    else:
        return(0) #if we did not find any diagonal lines, return an entropy of zero

def get_linemax(line_lengths):
    #input: a dictionary with all line lengths as key and the amount of lines of
    #       that length as value
    #output:the longest key of that dictionary
    all_lengths = list(line_lengths.keys())
    linemax = max(all_lengths)
    return(linemax)

def get_rqa_measures(cosine_sim_matrix, filename):
    #input: the cosine similarity matrix
    #calculates %REC, %DET, %LAM and linemax
    #fills the RQA_MEASURES dictionary
    #outputs nothing
    n_words = cosine_sim_matrix.shape[0]
    n_points_total = (n_words * n_words) - n_words
    if n_points_total < 1:
        print(filename, 'does not seem to contain enough words. Something went wrong. Not calculating any metrics for', filename)
        return
    measures = {}
    measures['length'] = n_words

    #%REC = [n recurrent points in matrix] / [n possible points in matrix]
    n_recurrent = get_recurrency(cosine_sim_matrix, n_words)
    rec_rate = n_recurrent/n_points_total
    measures['rec'] = rec_rate

    #%DET = [n recurrent points on a diagonal line] / [n recurrent points in matrix]
    n_on_diag = get_determinism(cosine_sim_matrix, n_words)
    if n_recurrent > 0:
        determinism = n_on_diag/n_recurrent
    else:
        determinism = 0
    measures['determinism'] = determinism

    #%LAM = [n recurrent points on a vertical line] / [n recurrent points in matrix]
    n_on_vert = get_laminarity(cosine_sim_matrix, n_words)
    if n_recurrent > 0:
        laminarity = n_on_vert/n_recurrent
    else:
        laminarity = 0
    measures['laminarity'] = laminarity

    #entropy of the length of diagonal line segments
    line_lengths = get_line_lengths(cosine_sim_matrix, n_words)
    line_length_entropy = get_line_length_entropy(line_lengths, n_words)
    measures['entropy'] = line_length_entropy

    #linemax is the length of the longest diagonal line
    linemax = get_linemax(line_lengths)
    measures['linemax'] = linemax

    RQA_MEASURES[filename] = measures

def get_rqa_measures_and_plot(embedding_vector, filename):
    cosine_sim_matrix = get_cosine_sim_matrix(embedding_vector)
    #create_recurrence_plot(cosine_sim_matrix, filename)
    get_rqa_measures(cosine_sim_matrix, filename)

###################################### PRINT ###################################

def print_measures():
    outputfile = open('rqa_measures_dreams_VALIDATED.csv', mode='w')
    i = 0
    for filename in RQA_MEASURES.keys():
        if i == 0: #if we are at the first iteration, print the headers
            outputfile.write('filename')
            for measure in RQA_MEASURES[filename].keys():
                outputfile.write(',')
                outputfile.write(measure)
            outputfile.write('\n')
        #dream_name = re.sub('\.tok\.txt', '', filename)
        dream_name = filename
        outputfile.write(dream_name)
        for measure in RQA_MEASURES[filename].keys():
            outputfile.write(',')
            value = str(RQA_MEASURES[filename][measure])
            outputfile.write(value)
        outputfile.write('\n')
        i += 1

    outputfile.close()

####################################### MAIN ###################################

def main(wdir='/home/hp/Documenten/labrot I', input_data_path = '/home/hp/Documenten/labrot I/dreams_validation'):
    #input_data_path is the path to the folder with the dreams (every dream is a separate .txt file)
    #wdir = '/home/hp/Documenten/labrot I'
    os.chdir(wdir) #make sure there is a map called 'plots' in your working directory!

    #All dream descriptions should have the name 'QQQ[number].tok.txt', so for example 'QQQ234.tok.txt'

    i = 0

    for file in os.listdir(input_data_path):
        path_to_file = input_data_path + '/' + file
        with open(path_to_file, 'r') as raw_dream:
            txt = preprocess_dream(raw_dream)
            lemma_list = create_lemma_list(txt)
            embedding_vector = get_embeddings(lemma_list)
            get_rqa_measures_and_plot(embedding_vector, file)
        i += 1
        if i % 100 == 0 or i == 1:
            print(i, " dreams out of ", len(os.listdir(input_data_path)), " analyzed")

    print_measures()

if __name__ == '__main__':
    main()
