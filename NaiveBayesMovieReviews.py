import os
import re
import math
import numpy
import time


# Decorator for timing function calls
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '    %r,  %2.2f sec' % \
              (method.__name__, te-ts)
        return result
    return timed


# Function to return cleaned, relevant words from a review
def returnWordsFromFile(file):
    p = re.compile(r'[a-zA-z]+')
    ch = '`^_[]'
    fid = open(file, 'r')
    words = p.findall(fid.read())
    words = [x.lower() for x in words]
    words = [x.translate(None, ch) for x in words]
    words = [w for w in words if len(w) > 1]
    fid.close()
    return words


# Function for counting occurrences of words for multinomial model
@timeit
def buildLikelihoodDict(directory, files):
    word_dict = {}
    total_count = 0
    for item in files:
        words = returnWordsFromFile(os.path.join(directory, item))
        total_count += len(words)
        for w in words:
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
    return word_dict, total_count


# Function for creating binary term matrix for Bernoulli model
@timeit
def buildBinaryTermArray(directory, files, V):
    # Initialize binary term array
    btarray = numpy.zeros(shape=(len(files), len(V)))
    for i in range(len(files)):
        words = returnWordsFromFile(os.path.join(directory, files[i]))
        # Populate binary term array with 1's where specific words occur
        for w in words:
            j = V[w]  # Grab column associated with current word
            btarray[i, j] = 1  # Populate with a 1
    return btarray


# Function to compute posterior probability for multinomial model
@timeit
def computePosteriorMultinomial(directory, files, priorPos, priorNeg,
                pos_word_dict, pos_word_count, neg_word_dict, neg_word_count, lenV):
    pos_total = 0
    neg_total = 0
    for item in files:
        # Start with prior probabilities
        pos_prob = math.log(priorPos)
        neg_prob = math.log(priorNeg)
        words = returnWordsFromFile(os.path.join(directory, item))
        for w in words:
            # Compute likelihood of word occurring in a positive review
            pos = float(pos_word_dict.get(w, 0) + 1) / (pos_word_count + lenV)
            pos_prob += math.log(pos)
            # Compute likelihood of word occurring in a negative review
            neg = float(neg_word_dict.get(w, 0) + 1) / (neg_word_count + lenV)
            neg_prob += math.log(neg)
        # Larger number (both will be negative) is 'winner'
        if pos_prob > neg_prob:
            pos_total += 1
        else:
            neg_total += 1

    # Calculate percentage of reviews classified as positive/negative
    percentPos = float(pos_total) / len(files)
    percentNeg = float(neg_total) / len(files)

    return percentPos, percentNeg


# Function to compute posterior probability for Bernoulli model
@timeit
def computerPosteriorBernoulli(directory, files, priorPos, priorNeg, pos_binary_sum, neg_binary_sum, V):
    pos_total = 0
    neg_total = 0
    for item in files:
        # Start with prior probabilities
        pos_prob = math.log(priorPos)
        neg_prob = math.log(priorNeg)
        # Initialize binary term array
        btarray = numpy.zeros(len(V))

        words = returnWordsFromFile(os.path.join(directory, item))
        for w in words:
            if w in V:
                j = V[w]
                btarray[j] = 1

        # Compute positive likelihood using vectorization
        term1 = numpy.power(pos_binary_sum, btarray)
        term2 = numpy.power(1-pos_binary_sum, 1-btarray)
        pos_like = numpy.multiply(term1, term2)
        pos_like_log = numpy.log(pos_like)
        pos_prob += numpy.sum(pos_like_log)

        # Compute negative likelihood using vectorization
        term1 = numpy.power(neg_binary_sum, btarray)
        term2 = numpy.power(1-neg_binary_sum, 1-btarray)
        neg_like = numpy.multiply(term1, term2)
        neg_like_log = numpy.log(neg_like)
        neg_prob += numpy.sum(neg_like_log)

        # Larger number (both will be negative) is 'winner'
        if pos_prob > neg_prob:
            pos_total += 1
        else:
            neg_total += 1

    # Calculate percentage of reviews classified as positive/negative

    percentPos = float(pos_total) / len(files)
    percentNeg = float(neg_total) / len(files)

    return percentPos, percentNeg


# Program start

pos_train_directory = '/Users/gdevore21/Documents/Projects/Independent/Movie Reviews/train/pos'
pos_files_train = os.listdir(pos_train_directory)
print('Using %i positive training reviews' % len(pos_files_train))

neg_train_directory = '/Users/gdevore21/Documents/Projects/Independent/Movie Reviews/train/neg'
neg_files_train = os.listdir(neg_train_directory)
print('Using %i negative training reviews' % len(neg_files_train))

# Calculate prior probabilities (percentage of pos/neg reviews)
priorPos = float(len(pos_files_train))/(len(pos_files_train) + len(neg_files_train))
priorNeg = float(len(neg_files_train))/(len(pos_files_train) + len(neg_files_train))

print
print('Prior probability for positive reviews = %f' % priorPos)
print('Prior probability for negative reviews = %f' % priorNeg)

# Read reviews
print('Reading positive reviews...')
(pos_word_dict, pos_word_count) = buildLikelihoodDict(pos_train_directory, pos_files_train)
print('Done, %i unique words logged, %i words total' % (len(pos_word_dict.keys()), pos_word_count))
print('Reading negative reviews...')
(neg_word_dict, neg_word_count) = buildLikelihoodDict(neg_train_directory, neg_files_train)
print('Done, %i unique words logged, %i words total' % (len(neg_word_dict.keys()), neg_word_count))

# Create V, unique list of all words in all reviews
print('Creating vocabulary vector V...')
combined_words = pos_word_dict.copy()
combined_words.update(neg_word_dict)
V = combined_words.keys()
V.sort()
print('There are %i unique words in all reviews' % len(V))
# Note position of ith word using a dictionary for population of binary vector
Vdict = {}
for i in range(len(V)):
    Vdict[V[i]] = i

# Create binary matrix for word occurrence in positive/negative reviews
print('Creating positive binary term matrix...')
pos_binary = buildBinaryTermArray(pos_train_directory, pos_files_train, Vdict)
# Calculate sums (number of documents containing each term)
pos_binary_sum = (numpy.sum(pos_binary, axis=0) + 1) / (len(pos_files_train) + 2)
print('Creating negative binary term matrix...')
neg_binary = buildBinaryTermArray(neg_train_directory, neg_files_train, Vdict)
# Calculate sums (number of documents containing each term)
neg_binary_sum = (numpy.sum(neg_binary, axis=0) + 1) / (len(neg_files_train) + 2)

# Test on new reviews
pos_test_directory = '/Users/gdevore21/Documents/Projects/Independent/Movie Reviews/test/pos'
pos_files_test = os.listdir(pos_test_directory)
print('Using %i positive testing reviews' % len(pos_files_test))

neg_test_directory = '/Users/gdevore21/Documents/Projects/Independent/Movie Reviews/test/neg'
neg_files_test = os.listdir(neg_test_directory)
print('Using %i negative testing reviews' % len(neg_files_test))

print('Testing reviews using multinomial model...')
# Positive reviews
(percentPos, percentNeg) = computePosteriorMultinomial(pos_test_directory, pos_files_test, priorPos, priorNeg,
                                        pos_word_dict, pos_word_count, neg_word_dict, neg_word_count, len(V))
multTP = percentPos  # Percent positive reviews
multFN = percentNeg  # Percent negative reviews
# Negative reviews
(percentPos, percentNeg) = computePosteriorMultinomial(neg_test_directory, neg_files_test, priorPos, priorNeg,
                                        pos_word_dict, pos_word_count, neg_word_dict, neg_word_count, len(V))
multTN = percentNeg  # Percent negative reviews
multFP = percentPos  # Percent positive reviews

# Calculate accuracy, sensitivity and specificity
multAcc = (multTP*len(pos_files_test) + multTN*len(neg_files_test))/(len(pos_files_test) + len(neg_files_test))
multSen = (multTP*len(pos_files_test))/(multTP*len(pos_files_test) + multFN*len(pos_files_test))
multSpc = (multTN*len(neg_files_test))/(multTN*len(neg_files_test) + multFP*len(neg_files_test))
print('Multinomial Accuracy: %f' % multAcc)
print('Multinomial Sensitivity: %f' % multSen)
print('Multinomial Specificity: %f' % multSpc)


print('Testing reviews using Bernoulli model...')
# Positive reviews
(percentPos, percentNeg) = computerPosteriorBernoulli(pos_test_directory, pos_files_test, priorPos, priorNeg,
                                        pos_binary_sum, neg_binary_sum, Vdict)
bernTP = percentPos  # Percent positive reviews
bernFN = percentNeg  # Percent negative reviews
# Negative reviews
(percentPos, percentNeg) = computerPosteriorBernoulli(neg_test_directory, neg_files_test, priorPos, priorNeg,
                                        pos_binary_sum, neg_binary_sum, Vdict)
bernTN = percentNeg  # Percent negative reviews
bernFP = percentPos  # Percent positive reviews

# Calculate accuracy, sensitivity and specificity
bernAcc = (bernTP*len(pos_files_test) + bernTN*len(neg_files_test))/(len(pos_files_test) + len(neg_files_test))
bernSen = (bernTP*len(pos_files_test))/(bernTP*len(pos_files_test) + bernFN*len(pos_files_test))
bernSpc = (bernTN*len(neg_files_test))/(bernTN*len(neg_files_test) + bernFP*len(neg_files_test))
print('Bernoulli Accuracy: %f' % bernAcc)
print('Bernoulli Sensitivity: %f' % bernSen)
print('Bernoulli Specificity: %f' % bernSpc)
