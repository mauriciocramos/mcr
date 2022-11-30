#!/bin/bash

#Speech to text batch script
#Parameters:
#    Input audio file path: /audios/*/*.mp3
#    Sample rate (hz): 16000 (default), 22050, 44100
#    Model path selection regex: /data/asr/models/vosk/*en-us* (for English models)
#    Model path name extraction regex: /data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$ (shorter names)
#    Output text file base path: /transcriptions/
#    Output sub-folder name extraction regex: /data/calls/audios(.*)/.mp3$ (get input subfolderts to output)

# All english models
python stt.py /data/calls/audios/voicemail/*.mp3 16000 /data/asr/models/vosk/*en-us* "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"
python stt.py /data/calls/audios/recording/*.mp3 16000 /data/asr/models/vosk/*en-us* "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"

# Only model vosk-model-en-us-aspire-0.2
python stt.py /data/calls/audios/voicemail/*.mp3 16000 /data/asr/models/vosk/vosk-model-en-us-aspire-0.2 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"
python stt.py /data/calls/audios/recording/*.mp3 16000 /data/asr/models/vosk/vosk-model-en-us-aspire-0.2 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$ "/data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"

# Only model vosk-model-en-us-0.22
python stt.py /data/calls/audios/voicemail/*.mp3 16000 /data/asr/models/vosk/vosk-model-en-us-0.22 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"
python stt.py /data/calls/audios/recording/*.mp3 16000 /data/asr/models/vosk/vosk-model-en-us-0.22 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"

# Only model vosk-model-small-en-us-0.15
python stt.py /data/calls/audios/voicemail/*.mp3 16000 /data/asr/models/vosk/vosk-model-small-en-us-0.15 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$ "/data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"
python stt.py /data/calls/audios/recording/*.mp3 16000 /data/asr/models/vosk/vosk-model-small-en-us-0.15 "/data/asr/models/vosk[/\\]vosk-model-(.*en-us.*)$" /data/calls/transcriptions/ "/data/calls/audios(.*)\.mp3$"
