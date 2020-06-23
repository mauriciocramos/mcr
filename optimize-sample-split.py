# Custom modules: add the 'src' directory as one where we can import modules
import os
import sys
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.insert(1, src_dir)
from model_selection.multilabel import multilabel_sample_dataframe, multilabel_train_test_split


# Standard modules
import pickle
import numpy as np
import pandas as pd


df = pd.read_csv('../data/TrainingData.csv', index_col=0)
LABELS = ['Function', 'Object_Type', 'Operating_Status', 'Position_Type', 'Pre_K', 'Reporting', 'Sharing',
          'Student_Type', 'Use']
NON_LABELS = list(set(df.columns) - set(LABELS))
df[LABELS] = df[LABELS].apply(lambda x: x.astype('category'), axis=0)
y = pd.get_dummies(df[LABELS], prefix_sep='__')
results = pd.DataFrame(columns=['sample_size', 'sample_min_count', 'sample_lessfreq', 'sample_ratio',
                                'test_size', 'test_min_count', 'train_lessfreq', 'test_lessfreq', 'train_ratio',
                                'test_ratio'])
y_mean = y.mean()
# The population less frequent class limits the sampling min_count otherwise it fails
LESS_FREQUENT_CLASS = y.sum().min()
for SAMPLE_SIZE in np.concatenate([np.linspace(0.01, 0.09, 9), np.linspace(0.1, 1, 10)]):
    for SAMPLE_MIN_COUNT in range(0, LESS_FREQUENT_CLASS + 1):
        sampling = multilabel_sample_dataframe(df, y, size=SAMPLE_SIZE, min_count=SAMPLE_MIN_COUNT, seed=1)
        dummy_labels = pd.get_dummies(sampling[LABELS], prefix_sep='__')
        sample_ratio = (dummy_labels.mean() / y_mean).mean()
        # The sample less frequent class limits the split min_count otherwise it fails
        SAMPLE_LESS_FREQUENT_CLASS = dummy_labels.sum().min()
        for TEST_SIZE in np.linspace(0.1, 0.9, 9):
            for TEST_MIN_COUNT in range(0, SAMPLE_LESS_FREQUENT_CLASS + 1):
                X_train, X_test, y_train, y_test = multilabel_train_test_split(sampling[NON_LABELS],
                                                                               dummy_labels,
                                                                               size=TEST_SIZE,
                                                                               min_count=TEST_MIN_COUNT,
                                                                               seed=1)
                train_ratio = (y_train.mean() / y_mean).mean()
                test_ratio = (y_test.mean() / y_mean).mean()
                train_lessfreq = y_train.sum().min()
                test_lessfreq = y_test.sum().min()
                msg = 'SAMPLE {:.2f} min {} lessfreq {} ratio {:.5f} SPLIT {:.1f} min {} LESSFREQ train {} test {} ' \
                      'RATIOS train {:.5f} test {:.5f}'
                print(msg.format(SAMPLE_SIZE, SAMPLE_MIN_COUNT, SAMPLE_LESS_FREQUENT_CLASS, sample_ratio, TEST_SIZE,
                                 TEST_MIN_COUNT, train_lessfreq, test_lessfreq, train_ratio, test_ratio))
                results = results.append({'sample_size': SAMPLE_SIZE,
                                          'sample_min_count': SAMPLE_MIN_COUNT,
                                          'sample_lessfreq': SAMPLE_LESS_FREQUENT_CLASS,
                                          'sample_ratio': sample_ratio,
                                          'test_size': TEST_SIZE,
                                          'test_min_count': TEST_MIN_COUNT,
                                          'train_lessfreq': train_lessfreq,
                                          'test_lessfreq': test_lessfreq,
                                          'train_ratio': train_ratio,
                                          'test_ratio': test_ratio}, ignore_index=True)
with open('../data/optimize-sample-split.pkl', 'wb') as file:
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)
