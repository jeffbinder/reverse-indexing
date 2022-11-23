from nltk.tokenize import sent_tokenize
import os
import pickle

with open('adamsmith-index.pkl', 'rb') as f:
    index = pickle.load(f)

page_headings = {}
for heading in index:
    for subheading, pgnum in index[heading]:
        if pgnum in page_headings:
            page_headings[pgnum].append((heading, subheading))
        else:
            page_headings[pgnum] = [(heading, subheading)]

with open('training-data.txt', 'w') as outf:
    last_sentence = None
    for fname in os.listdir('WoN1852-pages'):
        pgnum = int(fname)
        with open(os.path.join('WoN1852-pages', fname), 'r') as inf:
            pgtext = inf.read()
        pgtext = pgtext.replace('\n\n', '<p>')
        pgtext = pgtext.replace('\n', ' ')
        pgtext = pgtext.replace('<p>', '\n\n')
        training_text = '\n\n<|endoftext|>\n\n[Page headings:'
        training_text += '; '.join(
            f' {heading}, {subheading}'
            for heading, subheading in page_headings.get(pgnum, [])
        )
        if last_sentence is not None:
            training_text += f']\n\nTHE WEALTH OF NATIONS, PG {pgnum}\n\n{last_sentence} {pgtext}'
        else:
            training_text += f']\n\nTHE WEALTH OF NATIONS, PG {pgnum}\n\n{pgtext}'
        outf.write(training_text)
        last_sentence = sent_tokenize(pgtext)[-1]
    outf.write('<|endoftext|>')
