# Reverse Indexing: Training a computer to generate a book from its index

This repo contains code for my 2022 [entry](https://github.com/NaNoGenMo/2022/issues/33)
in [National Novel Generation Month](https://nanogenmo.github.io/), a challenge to write a computer program that generates a novel of at least 50,000 words.

My idea is reverse indexing: training language model to generate a book from its index. As training data, I used the index from Adam Smith's _The Wealth of Nations_, which I already digitized for an [earlier project](https://github.com/jeffbinder/adamsmith). I used this to finetune a version of GPT-2-xl, such that it can generate a page given its index headings.

As a first experiment after I finetuned the model, I tried getting it to reconstruct Smith's book from the index; you can see the results in the file "[the-wealth-of-nations-generated.txt](https://github.com/jeffbinder/reverse-indexing/blob/main/the-wealth-of-nations-generated.txt)."

For my actual entry, I will be generating a novel based on [this index](https://github.com/jeffbinder/reverse-indexing/blob/main/index.txt), which I wrote myself. I used [PromptArray](https://github.com/jeffbinder/promptarray), a prompting language I developed, to induce the generator to write less like Adam Smith and more like a novel.

The index I wrote is based loosely Joseph Campbell's account, in _The Hero with a Thousand Faces_ of the monomyth: a structure that (he claimed) is common to heroic narratives in many cultures. It's a somewhat worn-out idea, but I picked it for precisely that reason. Text generators are in some way chewing up and regurgitating the texts in the training data, so I wanted to generate a story that foregrounds its debt to old conventions for what a story should contain.

I also took inspiration from a passage in Michel Foucault's _The Order of Things_, in which he points out that some early-modern bestiaries mixed together facts about what we would now consider to be natural qualities of animals (size, color, shape) with facts about the values humans assign to them (role in mythology, heraldic meaning), drawing no fundamental distinction between the two.

I wanted to generate a text that, similarly, draws no clear line between narration of a story and critical commentary about that story. The entries I wrote for major characters therefore promise not just descriptions and actions, but also comments about the characters' resemblance to various cultural sources, criticisms of how the characters are depicted, and accusations of plagiarism. I am aiming not so much at self-reflexive metafiction as a style that declines to draw any line between fiction and metafiction.

The [initial results](https://github.com/jeffbinder/reverse-indexing/blob/main/tomhero-v1.txt) are somewhat rocky, and I am still working on the generation procedure. More to come...

If you want to run this code, you will need to download a copy of the [PromptArray](https://github.com/jeffbinder/promptarray) in this directory.