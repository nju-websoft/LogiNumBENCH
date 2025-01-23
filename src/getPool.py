import json
import re
from nltk.corpus import treebank, wordnet as wn, names, reuters
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

# Step 1: get relationPool
lemmatizer = WordNetLemmatizer()
transitive_verbs = set()


def is_word(token):
    return isinstance(token, str) and token.isalpha()


for tree in treebank.parsed_sents():
    for subtree in tree.subtrees(lambda t: t.label() == 'VP'):
        children = list(subtree)
        for i, child in enumerate(children):
            if child.label().startswith('VB'):
                verb_tokens = child.leaves()
                verb_tokens = [
                    token for token in verb_tokens if is_word(token)
                ]
                for verb in verb_tokens:
                    if i + 1 < len(children) and children[i +
                                                          1].label() == 'NP':
                        # 词形还原并转换为小写
                        lemma = lemmatizer.lemmatize(verb.lower(), pos=wn.VERB)
                        transitive_verbs.add(lemma)

transitive_verbs_list = list(transitive_verbs)

# Step 2: get entityPool
male_names = names.words('male.txt')
female_names = names.words('female.txt')
all_names = male_names + female_names

# Step 3: get attrPool

filtered_adjectives = [
    word for synset in wn.all_synsets('a') for word in synset.lemma_names()
    if len(word) > 1 and re.match("^[a-zA-Z]+$", word)
]

reuters_words = [
    word.lower() for fileid in reuters.fileids()
    for word in reuters.words(fileid)
]
fdist = FreqDist(reuters_words)

threshold = 10
high_freq_adjectives = [
    word.lower() for word in filtered_adjectives
    if fdist[word.lower()] > threshold
]

adjectives = list(set(high_freq_adjectives))

# Step 4: store pools
data_dict = {
    "relationPool": transitive_verbs_list,
    "entityPool": all_names,
    "attrPool": adjectives
}

with open('./configs/template-config.json', 'r') as json_file:
    data = json.load(json_file)
data |= data_dict

data["count"] = {
    key: len(value)
    for key, value in data.items() if isinstance(value, list)
}

with open('./configs/config.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("data has been saved to config.json")
