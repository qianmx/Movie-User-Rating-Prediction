import argparse
import re
import os
import csv
import math
import collections as coll
import pandas as pd
import numpy as np


def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries

    Input: filename

    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    # Your code here
    with open(filename, 'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            movieid = int(row[0])
            customerid = int(row[1])
            rating = float(row[2])
            if not user_ratings.has_key(customerid):
                user_ratings[customerid] = {}
            user_ratings[customerid][movieid] = rating
            if not movie_ratings.has_key(movieid):
                movie_ratings[movieid] = {}
            movie_ratings[movieid][customerid] = rating
    return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings

    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = {}
    # Your code here
    for i in user_ratings:
        ratings = user_ratings[i].values()
        avg = float(sum(ratings)) / float(len(ratings))
        ave_ratings[i] = avg
    return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users

        Input: d1, d2, (dictionary of user ratings per user)
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    # Your code here
    overlap = set(d1) & set(d2)
    if bool(overlap):
        numerator = 0
        d1_dev = 0
        d2_dev = 0
        for i in overlap:
            numerator += (d1[i] - ave_rat1) * (d2[i] - ave_rat2)
            d1_dev += (d1[i] - ave_rat1) ** 2
            d2_dev += (d2[i] - ave_rat2) ** 2
        denominator = math.sqrt(d1_dev * d2_dev)
        if denominator != 0:
            w = numerator / denominator
            return w
        else:
            return 0
    else:
        return 0.0


def main():
    """
    This function is called from the command line via

    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    # print train_file, test_file

    # your code here
    # get training set
    user_ratings_train, movie_ratings_train = parse_file(train_file)
    avg_rating_train = compute_average_user_ratings(user_ratings_train)
    avg_tot = sum(avg_rating_train.values())/len(avg_rating_train.values())

    def predict(row):
        k = int(row[0])  # movieid
        i = int(row[1])  # userid
        numerator = 0
        denominator = 0

        if i in user_ratings_train:
            d1 = user_ratings_train[i]
            ave_rat1 = avg_rating_train[i]
            if k in movie_ratings_train:  # a known user and a known movie
                for j in movie_ratings_train[k]:
                    d2 = user_ratings_train[j]
                    ave_rat2 = avg_rating_train[j]
                    w = compute_user_similarity(d1, d2, ave_rat1, ave_rat2)
                    denominator += abs(w)
                    if denominator > 0:
                        numerator += w * (user_ratings_train[j][k] - ave_rat2)
                        second_term = numerator / denominator
                    else:
                        second_term = 0
            else:  # a new movie and a known user
                second_term = 0
            r_pred = ave_rat1 + second_term
            return r_pred
        elif k in movie_ratings_train:  # a new user but a known movie
            ratings = movie_ratings_train[k].values()
            movie_avg = float(sum(ratings)) / float(len(ratings))
            return movie_avg
        else:
            return avg_tot  # a new user and new movie

    # test set
    test_data = pd.read_csv(test_file, header=None)
    test_data[3] = test_data.apply(predict, 1)

    prediction = test_data[3]
    actual = test_data[2]
    n = len(prediction)

    RMSE = round(np.sqrt((((prediction -actual)**2).sum())/n), 4)
    MAE = round(((abs(prediction - actual)).sum())/n, 4)

    # write to file and print result
    test_data.to_csv('predictions.txt', header=None, index=0)

    print 'RMSE', RMSE
    print 'MAE', MAE

if __name__ == '__main__':
    main()




