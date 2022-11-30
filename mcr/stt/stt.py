import os

from vosk import Model, KaldiRecognizer, SetLogLevel
from os import path
from subprocess import Popen, PIPE
from json import loads
from re import match
from time import time
from datetime import datetime
from glob import glob
import sys


def recognizer(input_path, model_path, sample_rate=16000):
    SetLogLevel(1)
    if not path.exists(model_path):
        print(f'missing model {model_path}')
        return
    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)
    process = Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                     input_path,
                     '-ar', str(sample_rate),  # set audio sampling rate (in Hz)
                     '-ac', '1',               # set number of audio channels
                     '-f', 's16le',            # force format s16le
                     '-'],
                    stdout=PIPE)
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            text = loads(rec.Result())['text']
            if text != '':
                yield text
                # print('result "{}"'.format(loads(rec.Result())['text']))
        # else:
            # print('partial {}'.format(loads(rec.PartialResult())))
    # print('final "{}"'.format(loads(rec.FinalResult())['text']))
    text = loads(rec.FinalResult())['text']
    if text != '':
        yield text


def batch_recognizer(source_paths=None, sample_rate=16000, model_paths=None, model_pattern=None, target_base_path=None,
                     target_pattern=None):
    for model_path in model_paths:
        for source_path in source_paths:
            model_name = f'{match(model_pattern, model_path).group(1)}-{sample_rate}hz'
            target_complement = f'{match(target_pattern, source_path).group(1)}.txt'
            target_path = f'{target_base_path}{model_name}{target_complement}'
            if path.isfile(target_path):
                print(f'skipping existing {target_path}')
            else:
                print(f'Transcribing {source_path} at {datetime.now():%Y-%m-%d %H:%M:%S}')
                t = time()
                try:
                    transcription = ' '.join(recognizer(source_path, model_path))
                except KeyboardInterrupt:
                    print(f'Transcribing interrupted {source_path} at {datetime.now():%Y-%m-%d %H:%M:%S}')
                    sys.exit(1)
                try:
                    with open(target_path, 'wt', encoding='utf-8') as f:
                        f.write(transcription)
                except BaseException as err:
                    print(f'{type(err).__name__}: {err}')
                    if os.path.isfile(target_path):
                        try:
                            os.remove(target_path)
                        except OSError as err:
                            print(f'{type(err).__name__}: {err}')
                    sys.exit(1)
                print(f'\rTranscribed {target_path} in {time()-t:0.1f} seconds at {datetime.now():%Y-%m-%d %H:%M:%S}')


if __name__ == '__main__':
    # Parameters:
    #    Input audio file path: /audios/*/*.mp3
    #    Sample rate (hz): 16000 (default), 22050, 44100
    #    Model path selection regex: /data/asr/models/vosk/*en-us* (for English models)
    #    Model path name extraction regex: /data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$ (shorter names)
    #    Output text file base path: /transcriptions/
    #    Output sub-folder name extraction regex: /data/calls/audios(.*)/.mp3$ (get input sub-folders to output)
    batch_recognizer(source_paths=glob(sys.argv[1]),
                     sample_rate=int(sys.argv[2]),
                     model_paths=glob(sys.argv[3]),
                     model_pattern=sys.argv[4],
                     target_base_path=sys.argv[5],
                     target_pattern=sys.argv[6])
