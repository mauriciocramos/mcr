import sys
from warnings import warn
import numpy as np
import pandas as pd
from glob import glob
from os import path
import re


# def transcription_files_generator(file_pattern, id_pattern):
#     # search the file pattern and yields the id (extracted with the id pattern) and the file bytes
#     for file in glob(file_pattern):
#         with open(file) as f:
#             yield id_pattern.match(file).group(1), f.read()


# def transcription_list_generator(file_pattern, ids):
#     # search the file pattern and yields the id (filtered by a list of ids) and the file bytes
#     for i in ids:
#         if not path.isfile(file_pattern.format(i)):
#             warn(f'Missing file {file_pattern.format(i)}')
#         else:
#             with open(file_pattern.format(i)) as f:
#                 yield i, f.read()


def read_file_contents(file_pattern, ids=None):
    """
    Retrieves files from pattern and yields their id and content
    :param file_pattern: text file path that may contain {} placeholder to inject id list items
    :param ids:
    None: get all files with sequence index.
    List: get files matching id list.
    re: regex to extract ids from the file name
    :return: tuples containing id and content
    """
    if ids is None:
        i = 0
        for file in glob(file_pattern):
            with open(file) as f:
                i += 1
                yield i, f.read
    elif isinstance(ids, list):
        if '{}' not in file_pattern:
            raise ValueError('id list requires a file pattern containing a {} placeholder')
        for i in ids:
            if not path.isfile(file_pattern.format(i)):
                warn(f'Missing file {file_pattern.format(i)}')
            else:
                with open(file_pattern.format(i)) as f:
                    yield i, f.read()
    elif isinstance(ids, str):
        if not re.search(r'\(.*\)', ids):
            raise ValueError(r"id string should contain a regex capture group e.g. r'.*\(.*)\.txt'")
        ids = re.compile(ids)
        for file in glob(file_pattern):
            with open(file) as f:
                yield ids.match(file).group(1), f.read()


def read_file_contents_to_dataframe(file_pattern, ids=r'.*\\(.*)\.txt'):
    """
    Generates a dataframe of file ids and contents
    :param file_pattern: text file path that may contain {} placeholder to inject id list items
    :param ids:
    None: Generates sequence indices.
    List: Inject ids in the file pattern {} placeholder.
    String: regex to extract ids from the file pattern, which should contain a capture group e.g. r'.*\\(.*)\\.txt'
    :return: yields tuples containing id and content.
    """
    return pd.DataFrame(read_file_contents(file_pattern, ids), columns=['id', 'transcription'])\
        .replace({'transcription': r'^\s*$'}, np.nan, regex=True)\
        .set_index('id', verify_integrity=True).sort_index()


# def read_transcription_list(file_pattern, ids):
#     # Generates a dataframe containing ids matched from a list of ids and their transcriptions
#     return pd.DataFrame(transcription_list_generator(file_pattern, ids), columns=['id', 'transcription'])\
#         .replace({'transcription': r'^\s*$'}, np.nan, regex=True)\
#         .set_index('id', verify_integrity=True).sort_index()


def read_transcriptions(file):
    # reads the merged csv
    return pd.read_csv(file, dtype=str).set_index('id', verify_integrity=True)


if __name__ == '__main__':

    # Example-1:
    # /data/calls/transcriptions/en-us-aspire-0.2-16000hz/voicemail/*.txt
    # ".*\\(.*)\.txt"
    # /data/calls/transcriptions/voicemails.csv

    # Example-2:
    # /data/calls/transcriptions/en-us-aspire-0.2-16000hz/recording/*.txt
    # ".*\\(.*)\.txt"
    # /data/calls/transcriptions/recordings.csv

    read_file_contents_to_dataframe(sys.argv[1], sys.argv[2]).to_csv(sys.argv[3], line_terminator='\n')
