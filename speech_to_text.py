import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('PsDyGpxeanjUMx2oKWf2mEutf4isL51Rl6MeJk3u0C7z')
service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url('https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/c31a39a4-4be3-4df9-9744-e870f1e78fe3')

def speech_to_text(file_path):
    words_to_censor = []

    with open(file_path, 'rb') as audio_file:
        result = service.recognize(
            audio=audio_file,
            content_type='audio/wav',
            timestamps=True,
            word_confidence=True,
            profanity_filter= False).get_result()

    timestamps = result['results'][0]['alternatives'][0]['timestamps']
    transcript = result['results'][0]['alternatives'][0]['transcript']
    for timestamp in timestamps:
        word_info = {
            "word": timestamp[0],
            "start": timestamp[1],
            "end": timestamp[2]
        }
        words_to_censor.append(word_info)
    
    words_to_censor.append({'transcript': transcript})

    return words_to_censor