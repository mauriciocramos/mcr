import pandas as pd
import pymongo

from util.util import get_dict_keys


def get_collection_fields(collection):
    # Function to extract a list of collection fields
    this_collection_fields = []
    # builds a list of generator functions, one per document
    for document in collection.find({}):
        this_collection_fields += get_dict_keys(document)
    used = set()
    # extract unique fields from the list of generator functions
    this_collection_fields = \
        [field for field in this_collection_fields if field not in used and (used.add(field) or True)]
    return this_collection_fields


# def get_collection_fields2(collection):
#     # Function to extract a list of collection fields
#     thisCollectionFields = []
#     # builds a list of generator functions, one per document
#     for document in collection.find({}):
#         thisCollectionFields += get_dict_keys2(document)
#     used = set()
#     # extract unique fields from the list of generator functions
#     thisCollectionFields =[field for field in thisCollectionFields if field not in used and (used.add(field) or True)]
#     print('\nCollection {}'.format(collection.name))
#     print(thisCollectionFields)
#     return thisCollectionFields


def get_mongodb_collections_as_dataframe_list(url, db, collections):
    # generator function returning a list of dataframes
    with pymongo.MongoClient(url) as mongo_client:
        mongodb = mongo_client.get_database(db)
        for collectionName in collections:
            collection = mongodb.get_collection(collectionName)
            projection = {field: 1 for field in get_collection_fields(collection)}
            yield pd.json_normalize(list(collection.find({}, projection)))
    return


def get_mongodb_collection(url, db, collection):
    # returns a collection as a list of dicts
    with pymongo.MongoClient(url) as mongo_client:
        mongodb = mongo_client.get_database(db)
        collection = mongodb.get_collection(collection)
        projection = {field: 1 for field in get_collection_fields(collection)}
        # write a list of documents to a file
        # with open('{}.json'.format(collection.name), 'w', encoding='utf-8') as f:
        #     f.write('{}'.format(list(collection.find({}, projection))))
        return list(collection.find({}, projection))
