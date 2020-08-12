'''
Name: Vandit Gajjar
Task: Image Inalytics 
University: The University of Adelaide 
Submission file for AI Australia Task
File: Fetching the label
'''

import os

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    else:
        raise Exception('Invalid dataset!!!')