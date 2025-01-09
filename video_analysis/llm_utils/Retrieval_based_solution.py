#%%
import os
import openai
import yaml

import re


import json
import re

# Load API key from a YAML configuration file
with open("config.yaml") as f:
 config_yaml = yaml.load(f, Loader=yaml.FullLoader)

# Initialize the OpenAI client with the API key
client = openai.OpenAI(api_key=config_yaml['token'])

# Set the model name
MODEL = "gpt-4o-mini"

def llm(prompt, stop=["\n"]):
    # Prepare the dialog for the API request
    dialogs = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content

def get_single_sentence(original_action):
    
 
    prompt = f"""
     I want to refrase the action description "{original_action}" in to a simpler and clear and clean sentence. 
     The sentence should start with " A parson ... " and should be easy to understand and should only use words from the words list blow.
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
     
     The output can only be one sentence:" A parson ... " 
     The output can only be one sentence:" A parson ... " 
     do NOT add any other description!
     
     you should only use words from the words_list
     you should only use words from the words_list
     you should only use words from the words_list
     you should only use words from the words_list
    <END_SIGNAL>
    - When your sentence is finished, signal completion by saying '<DONE>'.
    </END_SIGNAL>
     
    """
    
    return prompt
    
def sequence_analyze(action):
    result = get_single_sentence(action)
    result1 = llm(result, stop=["<DONE>"]).split("<DONE>")[0].strip()
    print(result1)
    return result1

actions_to_test = [
    "The person performs a rowing motion with their legs spread wide.",
    "A woman hops forward while holding a T-pose.",
    "The man executes a jump spin mid-air.",
    "A person crawls on the ground in a baby-like motion.",
    "A dancer spins gracefully in a ballet twirl.",
    "The computer science student attempts one-armed push-ups.",
    "The soccer player covers their ears during a goal celebration.",
    "A dad crawls on the floor to retrieve his child’s toy.",
    "The police officer sprints to chase someone on foot.",
    "A bodybuilder performs a perfect pistol squat."
]

for action in actions_to_test:
    sequence_analyze(action)
    

