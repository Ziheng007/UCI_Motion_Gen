# %%
import os
import openai
import yaml

import re

import json
import re

# Load API key from a YAML configuration file
with open("/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/llm_utils/config.yaml") as f:
 config_yaml = yaml.load(f, Loader=yaml.FullLoader)

# Initialize the OpenAI client with the API key
client = openai.OpenAI(api_key=config_yaml['token'])

# Set the model name
MODEL = "gpt-4o"


def llm(prompt, stop=["\n"]):
    # Prepare the dialog for the API request
    dialogs = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content


def get_single_sentence(original_action,reasoning):
    prompt = f"""
     I want to refrase the action description "{original_action}" in to a simpler and clear and clean sentence. 
     you should see the import reasoning of the action
     
     <reasoning>
     original_action :"{original_action}"
     reasoning: "{reasoning}"
     </reasoning>
     
     The sentence should start with " a person ... " and should be easy to understand and should only use words from the words list blow.
     A word can be used as long as it is mentioned in the **words_list**, regardless of its form. For example, if "walking" is in the **words_list**, then "walks" can also be used.

     <words_list>
     words = [
    'the', 'to', 'a', 'then', 'their', 'right', 'his', 'forward', 'with', 'walks', 'left', 'in', 'is', 'up', 'arms', 
    'back', 'down', 'on', 'something', 'steps', 'walking', 'side', 'arm', 'both', 'around', 'it', 'of', 'while', 
    'turns', 'sting', 'at', 'from', 'sts', 'takes', 'slowly', 'leg', 'head', 'out', 'them', 'raises', 'as', 'moves', 
    'foot', 'front', 'an', 'circle', 'backwards', 'over', 'he', 'motion', 'figure', 'they', 'moving', 'position', 
    'before', 'jumps', 'again', 'step', 'clockwise', 'one', 'does', 'object', 'puts', 'slightly', 'someone', 'straight', 
    'lifts', 'place', 'stops', 'if', 'bends', 'body', 'ground', 'doing', 'sides', 'holding', 'by', 'picks', 'sits', 
    'runs', 'face', 'jumping', 'like', 'this', 'for', 'still', 'times', 'be', 'legs', 'few', 'knees', 'walk', 'forth', 
    'waves', 'above', 'towards', 'air', 'two', 'throws', 'holds', 'lowers', 'quickly', 'kicks', 'direction', 'makes', 
    'feet', 'appears', 'its', 'forwards', 'together', 'sitting', 'starts', 'gets', 'jogs', 'line', 'shoulder', 'looks', 
    'using', 'chest', 'bent', 'brings', 'running', 'into', 'swinging', 'swings', 'was', 'stretches', 'who', 'leans', 
    'raised', 'turn', 'reaches', 'twice', 'are', 'turning', 'counter', 'walked', 'stairs', 'floor', 'uses', 'waving', 
    'other', 'backward', 'squats', 'or', 'him', 'has', 'balance', 'several', 'some', 'starting', 'off', 'throwing', 
    'places', 'dancing', 'raising', 'knee', 'original', 'ball', 'making', 'jacks', 'hips', 'circular', 'same', 'lifting', 
    'across', 'time', 'stumbles', 'jump', 'stepping', 'taking', 'very', 'pace', 'returns', 'height', 'bending', 
    'jogging', 'looking', 'small', 'dance', 'hops', 'pick', 'put', 'spot', 'three', 'diagonally', 'seems', 'drops', 
    'waist', 'shoulders', 'begins', 'goes', 'move', 'away', 'once', 'facing', 'way', 'claps', 'pushed', 'after', 
    'stopping', 'continues', 'level', 'but', 'that', 'onto', 'sideways', 'circles', 'behind', 'motions', 'first', 
    'crosses', 'st', 'stretching', 'item', 'four', 'crawls', 'spins', 'putting', 'where', 'grabs', 'wide', 'high', 
    'shakes', 'another', 'large', 'fast', 'kicking', 'all', 'chair', 'elbow', 'counterclockwise', 'trying', 'diagonal', 
    'kick', 'punches', 'throw', 'more', 'slow', 'each', 'near', 'toward', 'leaning', 'elbows', 'dances', 'opposite', 
    'half', 'pattern', 'sit', 'touches', 'squatting', 'degrees', 'top', 'sets', 'carefully', 'rotates', 'room', 'paces', 
    'stick', 'upper', 'extended', 'started', 'crouches', 'apart', 'stance', 'pose', 'playing', 'her', 'stop', 'mouth', 
    'phone', 'sim', 'swaying', 'gestures', 'themselves', 'squat', 'repeatedly', 'end', 'outstretched', 'extends', 
    'shape', 'thing', '180', 'big', 'bend', 'going', 'repeats', 'five', 'little', 'take', 'path', 'there', 't', 'shaking', 
    'sways', 'shuffles', 'picking', 'start', 'getting', 'casually', 'center', 'wrist', 'himself', 'normal', 'catches', 
    'couple'， 'swing', 'drink', 'climbs', 'multiple', 'upwards', 'do', 'reaching', 'look', 'quick', '4', 
    'placing', 'seconds', 'performs', 'kneels', 'balancing', 'table', 'weight', 'grabbed', 'not', 'torso', 'boxing', 
    'hold', 'crawling', 'cross', 'crossing', 'slight', 'scratches', 'pushes', 'clapping', 'gesture', 'bit', 'rope', 
    'sidesteps', 'performing', 'bows', 'get', 'about', 'hopping', 'than', 'pivots', 'twists', 'did', 'swimming', 'angle', 
    'watch', 'proceeds', 'stretch', 'point', 'make', 'surface', 'hip', 'rubs', 'grab', 'short', 'foward', 'run', 'whilst', 
    'go', 'punch', 'sat', 'push', 'drinking', 'punching', 'lift', 'full', 'comes', 'touching', 'upward', 'bringing', 
    'pauses', 'ner', 'acting', 'fours', 'returning', 'toes', 'middle', 'crouching', 'rail', 'finally', 'wave', 'touch', 
    'kneeling', 'beam', 'bottle', 'basketball', 'shelf', 'low', 'area', 'were', 'being', 'outward', 'lowering', 
    'grabbing', 'things', 'golf', 'spread', 'support', 'seated', 'almost', 'turned', 'treadmill', 'rotating', 'gently', 
    'without', 'though', 'so', 'crossed', 'curve', 'held', 'wiping', 'stomach', 'spinning', 'chicken', 'hunched', 
    'along', 'speed', 'moved', 'opens', 'drinks', 'raise', 'wall', 'wipes', 'just', 'fighting', 'shifts', 'lower', 
    'neck', 'bottom', 'red', '5', 'upright', 'steady', 'degree', 'box', 'fall', 'movement', 'balances', 'catch', '90', 
    'directions', 'i', 'hit', 'ready', 'warm', 'guitar', 'washing', 'screen', 'pushing', 'bounces', 'stretched', 
    'through', 'crouched', 'long', 'fashion', 'jumped', 'wheel', 'reach', 'ahead', 'came', 'second', 'cartwheel', 
    'lightly', 'objects', 'jog', 'semi', 'exercise', 'return', 'forearm', 'staggers', 'medium', 'pulls', 'stumbling', 
    'twisting', 'railing', 'briskly', 'heavy', 'alternating', 'falls', 'took', 'different', 'open', 'coming', 
    'suddenly', 'hop', 'either', 'weights', 'camera', 'slides', 'close', 'zig', 'keeping', 'keeps', 'leading', 'resting', 
    'outwards', 'tennis', 'lunges', 'thighs', 'violin', 'trips', 'keep', 'extending', 'what', 'hrail', 'curved', 
    'eating', 'only', 'individual', 'bring', 'set', 'ear', 'ls', 'downward', 'square', 'skips', 'legged', 'briefly', 
    'moment', 'between', 'rubbing', 'robot', 'quarter', 'sprints', 'toon', 'movements', 'tilts', 'tries', 'door', 
    'rolls', 'pours', 'karate', 'plane', 'overhead', 'spin', 'arc', 'anti', 'zag', 'made', 'complete', 'scratching', 
    'salsa', 'clock', 'steering', 'action', 'warming', 'spreads', 'when', 'bow', 'pacing', 'pull', 'baseball', 'shake', 
    'platform', 'backs', 'exercises', 'lean', 'flaps', 'dog', 'directly', 'checks', 'items', 'ends', 'normally', 
    'crawl', 'blue', 'flapping', 'points', 'seat', 'until', 'which', 'clasps', 'dumbbell', 'cleaning', 'upstairs', 
    'climbing', 'come', 'acts', 'character', 'pretending', 'recovers', 'wiggles', 'continuing', 'higher', 'cup', 
    'laying', 'else', 'followed', 'faces', 'call', 'beginning', 'talking', 'help', 'pain', 'hard', 'eight', 'ending', 
    'relaxed', 'catching', 'picked', 'appearing', 'she', 'stool', 'loosely', 'subject', 'sweeping', 'football', 'shoe', 
    'ladder', 'length', 'rest', 'folds', 'dragging', 'distance', 'against', 'monkey', 'wise', 're', 'got', 'confidently', 
    'exercising', 'hitting', 'twist', 'rapidly', 'no', 'uneven', 'waltz', 'stood', 'hu', 'lets', 'ran', 'oval', 'ballet', 
    'cautiously', 'elevated', 'have', 'strides', 'defensive', 'anticlockwise', 'jack', 'six', 'dropping', 'somewhere', 
    'loses', 'repeat', 'leaps', 'having', 'simultaneously', 'ramp', 'moderate', 'gesturing', 'under', 'curls', 'avoid', 
    'jesture', '8', 'well', 'driving', 'completely', 'downwards', 'lifted', 'ling', 'obstacle', 'resumes', 
    'straightens', 'fist', 'your', 'crouch', 'downstairs', 'club', 'bouncing', 'bird', 'plays', 'horizontally', 
    'hanging', 'leaned', 'drunk', 'round', 'lays', 'location', 'hits', 'edge', 'rocking', 'fingers', 'marches', 
    'pretends', 'stirring', 'mixing', 'washes', 'pulling', 'also', 'style', 'shrugs', 'wipe', 'hed', 'forearms', 
    'limping'
]
     </words_list>

     The output can only be one sentence:" a person ... " 
     The output can only be one sentence:" a person ... " 
     do NOT add any other description!

     you should only use words from the words_list
 
    <END_SIGNAL>
    - When your sentence is finished, signal completion by saying '<DONE>'.
    </END_SIGNAL>

    """

    return prompt


def get_reasoning(original_action):
    prompt = f"""

    you are a reasoning and action model. your role is to, given a text description of a motion, first reason how the human body should look like, and then simplify the sentence into more universally understood body-part movement descriptions, so that anyone not familiar with the specifics of the phrase can understand the motion conatined in it. your reasoning step consists of understanding the motion, then synthesizing how the general motion pose looks like for the person, and then pinpoint how their arms and legs look like. It is essential that you keep the output sentence simple, even if the input motion is complicated, in which case you will do your best to encapsulate the key identifying points of the motion into, at most, two different motions within the output sentence. It is CRITICAL that you simplify the output to at most 2 separate motions, even if the originally described motion is a little less precise because of this.
    here is an example to get you started:
    motion: the worried contractor walks in a hurry
    reasoning 1: The prompt describes a person walking in a hurry. reasoning 2: How does a person look like when they are walking in a hurry? what are the main characteristics of the body when performing that action? reasoning 3: Walking in a hurry is walking with increased speed, so the body parts behave similar to a walk, that is, arms moving alternately, as well as legs.
    output: the man walks fast

    Now, I want to analyze the action description "{original_action}"  

    <END_SIGNAL>
    - When your sentence is finished, signal completion by saying '<DONE>'.
    </END_SIGNAL>

    """
    return prompt


# def get_single_sentence3(original_action):
#     prompt = f"""
#
#
#
#     you are a reasoning and action model. your role is to, given a text description of a motion, first reason how the human body should look like, and then simplify the sentence into more universally understood body-part movement descriptions, so that anyone not familiar with the specifics of the phrase can understand the motion conatined in it. your reasoning step consists of understanding the motion, then synthesizing how the general motion pose looks like for the person, and then pinpoint how their arms and legs look like. It is essential that you keep the output sentence simple, even if the input motion is complicated, in which case you will do your best to encapsulate the key identifying points of the motion into, at most, two different motions within the output sentence. It is CRITICAL that you simplify the output to at most 2 separate motions, even if the originally described motion is a little less precise because of this.
#     here is an example to get you started:
#     motion: the worried contractor walks in a hurry
#     reasoning 1: The prompt describes a person walking in a hurry. reasoning 2: How does a person look like when they are walking in a hurry? what are the main characteristics of the body when performing that action? reasoning 3: Walking in a hurry is walking with increased speed, so the body parts behave similar to a walk, that is, arms moving alternately, as well as legs.
#     output: the man walks fast
#
#     Now, I want to refrase the action description "{original_action}" in to a simpler and clear and clean sentence.
#      The sentence should start with " a person ... " and should be easy to understand.
#
#
#      The output can only be one sentence:" a person ... "
#      The output can only be one sentence:" a person ... "
#      do NOT add any other description!
#
#
#     <END_SIGNAL>
#     - When your sentence is finished, signal completion by saying '<DONE>'.
#     </END_SIGNAL>
#
#     """
#     return prompt
#

def first_sequence_analyze(action):
    get_reasoning_prompt = get_reasoning(action)
    reasoning = llm(get_reasoning_prompt, stop=["<DONE>"]).split("<DONE>")[0].strip()
    #print(reasoning)

    prompt = get_single_sentence(action,reasoning)
    result = llm(prompt, stop=["<DONE>"]).split("<DONE>")[0].strip()
    print(result)
    # print(result)
    return result


# actions_to_test = [
#     "The person performs a rowing motion with their legs spread wide.",
#     "A woman hops forward while holding a T-pose.",
#     "The man executes a jump spin mid-air.",
#     # "A person crawls on the ground in a baby-like motion.",
#     # "A dancer spins gracefully in a ballet twirl.",
#     # "The computer science student attempts one-armed push-ups.",
#     # "The soccer player covers their ears during a goal celebration.",
#     # "A dad crawls on the floor to retrieve his child’s toy.",
#     # "The police officer sprints to chase someone on foot.",
#     # "A bodybuilder performs a perfect pistol squat."
# ]

# for action in actions_to_test:
#     first_sequence_analyze(action)


