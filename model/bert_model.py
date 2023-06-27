from __future__ import print_function
import os
import json
import pickle
import numpy as np
import pandas as pd
import utils
import h5py
import torch
import time
from torch.utils.data import Dataset
from transformers import BertTokenizer

COUNTING_ONLY = False
dataroot = '/hpcwork/lect0099/data'
imgid_dataroot = dataroot+"/imgids"
adaptive = False

h5_dataroot = dataroot+"/Bottom-up-features-adaptive"\
if adaptive else dataroot+"/Bottom-up-features-fixed"
imgid_dataroot = dataroot+"/imgids"
name = 'train'
prefix = '36'

label2ans_path = os.path.join(dataroot, 'cache',
                                'trainval_label2ans.pkl')
label2ans = pickle.load(open(label2ans_path, 'rb'))

img_id2idx = pickle.load(
    open(os.path.join(imgid_dataroot, '%s%s_imgid2idx.pkl' %
                        (name, '' if adaptive else prefix)), 'rb'))

h5_path = os.path.join(h5_dataroot, '%s%s.hdf5' %
                        (name, '' if adaptive else prefix))

# Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False
    
def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
        (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            assert_eq(question['question_id'], answer['question_id'])
            assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))
    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries

start = time.time()
entries_test = _load_dataset(dataroot, name, img_id2idx, label2ans)
entries_df = pd.DataFrame(entries_test)
questions = entries_df.question.values
total_time = time.time() - start
print(total_time)




# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Print the original question.
print(' Original: ', questions[0])
# Print the question split into tokens.
print('Tokenized: ', tokenizer.tokenize(questions[0]))
# Print the question mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(questions[0])))



start_time_2 = time.time()
# max_len = 0

# # For every question...
# for ques in questions:

#     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#     input_ids = tokenizer.encode(ques, add_special_tokens=True)

#     # Update the maximum question length.
#     max_len = max(max_len, len(input_ids))

# print('Max question length: ', max_len)


# Tokenize all of the questions and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every question...
for ques in questions:
    # `encode_plus` will:
    #   (1) Tokenize the question.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the question to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        ques,                      # question to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all questions.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded question to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ques_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
# labels = torch.tensor(labels)

# Print question 0, now as a list of IDs.
print('Original: ', questions[0])
print('Token IDs:', input_ques_ids[0])
print('Attention masks: ', attention_masks[0])

print("time for encoding: ", time.time() - start_time_2)

from transformers import BertModel, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertModel.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = True, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

print()
