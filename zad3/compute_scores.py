import argparse
import json
import numpy as np

"""
==========================================
Movie recommendation system
==========================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to have argparse, json and numpy packages
To run script you need to provide username as parameter --user and json file with rated movies in the same folder

The system recommends movies based on lists of rated movies.
The program needs at least 4 user's movie list with rates.
Based on that the program will output 5 recommended movies and 5 movies not to watch.
* (Inputs)
   - `user`
      * User name of the user you want to find recommended movies
   - rating 
      * Json file name 'ratings.json' with rated movies lists.
      Example:
      {
        "Foo": 
        {
                "x": 10,
                "y": 10,
                "z": 8,
        },
        "Bar": 
        {
                "c": 10,
                "v": 10,
                "b": 10,
        }
      }
* (Output)
   - `Movie recommendation bases on Pearson algorithm`
      * Name and score of highest matching user
      * 5 movies to recommend and 5 movies not to recommend
   - `Movie recommendation bases on Euclidean algorithm`
      * Name and score of highest matching user
      * 5 movies to recommend and 5 movies not to recommend
"""


def build_arg_parser():
    """
        :return: passed by argument
    """
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True,
                        help='User')
    return parser


def euclidean_score(dataset, user1, user2):
    """
        :return: the Euclidean distance score between user1 and user2 based on provided data
    """
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users, 
    # then the score is 0 
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearson_score(dataset, user1, user2):
    """
        :return: the Pearson correlation score between user1 and user2 based on provided data
    """
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0 
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def getUserList(dataset):
    """
        :return: list of user names
    """
    userList = []
    for i in dataset:
        if i != user1:
            userList.append(i)
    return userList


def getMatchingResults(userList, data, mainUser):
    """
        Function calculates scores for every user using both algorithms
        :return: pearson and euclidean scores for every user
    """
    pearsonScoreList = {}
    euclideanScoreList = {}
    for user in userList:
        euclideanScoreList[user] = euclidean_score(data, mainUser, user)
        pearsonScoreList[user] = pearson_score(data, mainUser, user)
    pearsonScoreList = sorted(pearsonScoreList.items(), key=lambda x: x[1], reverse=True)
    euclideanScoreList = sorted(euclideanScoreList.items(), key=lambda x: x[1], reverse=True)
    return pearsonScoreList, euclideanScoreList


def getMoviesToRecommend(mainUser, matchingUser, data):
    """
        Function sorts movie list by rates descending
    """
    mainUserMovies = data[mainUser]
    matchingUserMovies = sorted(data[matchingUser].items(), key=lambda x: x[1], reverse=True)
    print("Recommended movies:")
    printMovies(matchingUserMovies, mainUserMovies)


def getMoviesNotToRecommend(mainUser, scoreList, data):
    """
        Function takes 3 highest matching user movie lists, sums it up
        If movie names are the same it takes movie rate from highest matching user
        Then it sorts movie list by rates ascending
    """
    mainUserMovies = data[mainUser]
    worstMoviesList = data[scoreList[2][0]]
    worstMoviesList.update(data[scoreList[1][0]])
    worstMoviesList.update(data[scoreList[0][0]])
    worstMoviesList = sorted(worstMoviesList.items(), key=lambda x: x[1], reverse=False)
    print("Do not watch this:")
    print("Based on worst rated movies by 3 best matching people")
    printMovies(worstMoviesList, mainUserMovies)


def printMovies(movieList, mainUserMovies):
    """
        Function prints 5 movies from list if the movie is not in the main User movie list
    """
    counter = 0
    for i in movieList:
        if i[0] not in mainUserMovies and counter < 5:
            print(i)
            counter += 1


def printUserRecommendInfo(user, matchingUser, score, type):
    """
        Function prints basic users info
    """
    print("Data gathered using %s algorithm" % type)
    print("The best matching user for %s is %s with score: %f" % (user, matchingUser, score))


if __name__ == '__main__':
    """
        Initialization of the program
    """
    args = build_arg_parser().parse_args()
    user1 = args.user

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    userList = getUserList(data)
    pearsonScoreList, euclideanScoreList = getMatchingResults(userList, data, user1)
    printUserRecommendInfo(user1, pearsonScoreList[0][0], pearsonScoreList[0][1], "Pearson")
    getMoviesToRecommend(user1, pearsonScoreList[0][0], data)
    getMoviesNotToRecommend(user1, pearsonScoreList, data)

    print("--------------------------------------------")

    printUserRecommendInfo(user1, euclideanScoreList[0][0], euclideanScoreList[0][1], "Euclidean")
    getMoviesToRecommend(user1, euclideanScoreList[0][0], data)
    getMoviesNotToRecommend(user1, euclideanScoreList, data)
