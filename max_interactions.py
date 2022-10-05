def max_interactions(n, minus=0):
    # Returns the number of features whose interactions don't exceed a maximum of n features minus the number of pre-existing features
    return int(((8 * n + 1)**(1/2) - 1) / 2) - minus
