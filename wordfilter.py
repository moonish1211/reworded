# %%
import nltk
from nltk.corpus import words

nltk.download('words')

eng = set(words.words())

with open("incorrect_advance.txt", "r+")as file:
    wrd = [line.strip() for line in file]
    wrd = [seq for seq in wrd if seq.lower() not in eng]

    file.seek(0)
    file.truncate()
    file.write("\n".join(wrd))

with open("incorrect_trick.txt", "r+")as file:
    wrd = [line.strip() for line in file]
    wrd = [seq for seq in wrd if seq.lower() not in eng]

    file.seek(0)
    file.truncate()
    file.write("\n".join(wrd))
