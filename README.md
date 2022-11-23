# Reverse Indexing: Training a computer to generate a book from its index

This repo contains code for my 2022 [entry](https://github.com/NaNoGenMo/2022/issues/33)
in [National Novel Generation Month](https://nanogenmo.github.io/), a challenge to write a computer program that generates a novel of at least 50,000 words.

My idea is reverse indexing: training language model to generate a book from its index. As training data, I used the index from Adam Smith's _The Wealth of Nations_, which I already digitized for an [earlier project](https://github.com/jeffbinder/adamsmith). I used this to finetune a version of GPT-2-xl, such that it can generate a page given its index headings.

As a first experiment after I finetuned the model, I tried getting it to reconstruct Smith's book from the index; you can see the results in the file "[the-wealth-of-nations-generated.txt](https://github.com/jeffbinder/reverse-indexing/blob/main/the-wealth-of-nations-generated.txt)."

More to come...

If you want to run this code, you will need to download a copy of the [PromptArray](https://github.com/jeffbinder/promptarray) in this directory.