# coding=utf-8
# Copyright (c) 2022 Jeffrey M. Binder.  All rights reserved.

import pickle
from nltk.tokenize import sent_tokenize

import sys
sys.path.append('promptarray')
from generation_utils import *

model_type = 'gpt2'
model_name_or_path = 'gpt2xl-finetuned-v2'
device = 'cuda'

generation_length = 500
do_sample = True
temperature = 0.6
k = 5
p = 0.5
repetition_penalty = 1.25
overlap_factor = 0.25

# Initialize the model and tokenizer
try:
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
model = model_class.from_pretrained(model_name_or_path)
model.to(device)
model.eval()

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = generation_length  # avoid infinite loop
    return length
length = adjust_length_to_model(generation_length, max_sequence_length=model.config.max_position_embeddings)

def generate_sequence(prompt_text):
    output_sequences = model.generate(
        prompt=prompt_text,
        overlap_factor=overlap_factor,
        tokenizer=tokenizer,
        min_length=10000,
        max_length=length,
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_return_sequences=1,
        pad_token_id=0,
        verbose=False,
        bad_words_ids = [
            tokenizer('[').input_ids,
            tokenizer('\n\n\n').input_ids,
            tokenizer('THE').input_ids,
        ]
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequence = output_sequences[0].tolist()
    generated_sequence = [idx for idx in generated_sequence if idx != 0]

    # Decode text
    generated_text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    return generated_text

def generate_page(title, pgnum, headings, prev_sentence, discourage=True):
    if discourage:
        prompt_text = f'''

{{[Page headings: {{{';  '.join(headings)}~Economics and manufactures}}]

{{{title}~THE WEALTH OF NATIONS}}, PG {pgnum}~}}

{prev_sentence}'''
    else:
        prompt_text = f'''

[Page headings: {';  '.join(headings)}]

{title}, PG {pgnum}

{prev_sentence}'''

    print(prompt_text)
    return generate_sequence(prompt_text)

def generate_book(title, index, filename, discourage=True):
    page_headings = {}
    for heading in index:
        for subheading, pgnum in index[heading]:
            if pgnum in page_headings:
                page_headings[pgnum].append((heading, subheading))
            else:
                page_headings[pgnum] = [(heading, subheading)]
    
    with open(filename, 'w') as f:
        last_sentence = ''
        for pgnum in sorted(int(x) for x in page_headings.keys()):
            f.write(f'{title}, PG {pgnum}\n\n')
            headings = [x[0] + ', ' + x[1] for x in page_headings[pgnum]]
            pgtext = generate_page(title, pgnum, headings, last_sentence, discourage)
            pgtext = pgtext.strip()
            f.write(f'{pgtext}\n\n')
            f.flush()
            last_sentence = sent_tokenize(pgtext)[-1]

def generate_the_wealth_of_nations():
    title = 'THE WEALTH OF NATIONS'
    with open('adamsmith-index.pkl', 'rb') as f:
        index = pickle.load(f)
    generate_book(title, index, 'the-wealth-of-nations-generated.txt', discourage=False)

def generate_from_index(title, index_filename, out_filename):
    index = {}
    with open(index_filename, 'r') as f:
        current_heading = None
        for line in f.readlines():
            if line.isspace():
                continue
            if line.startswith(' '):
                if current_heading is None:
                    raise Exception('Invalid indentation')
                subheading_list = []
                subheading, pgnums = line.strip().split(', ', 1)
                for pgnum in pgnums.split(', '):
                    if '-' in pgnum:
                        pgmin, pgmax = pgnum.split('-')
                        for i in range(int(pgmin), int(pgmax)+1):
                            subheading_list.append((subheading, i))
                    else:
                        subheading_list.append((subheading, int(pgnum)))
                if current_heading in index:
                    index[current_heading] += subheading_list
                else:
                    index[current_heading] = subheading_list
            else:
                current_heading = line.strip()

    generate_book(title, index, out_filename)

generate_from_index('THE JOURNEY OF TOM HERO: A NOVEL', 'index.txt', 'tomhero.txt')