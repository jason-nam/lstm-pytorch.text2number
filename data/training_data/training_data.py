import json
import os
rootdir = './data/training_data/labeling_data'

sent = []
tag = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print(os.path.join(subdir, file))

        with open(os.path.join(subdir, file), encoding="utf8") as file:
            sent.append(json.load(file)["script"]["scriptTN"])

inside = False
beginning = False

for i, s in enumerate(sent):
    sent_tag = ""
    for c in s:
        if c == "[":
            inside = True
            beginning = True
            continue
        elif c == "]":
            inside = False
            continue
        if inside:
            if beginning:
                sent_tag = sent_tag + "B"
                beginning = False
            else:
                sent_tag = sent_tag + "I"
        else:
            sent_tag = sent_tag + "O"
    tag.append(" ".join(sent_tag))
    s = s.replace("[", "")
    s = s.replace("]", "")
    sent[i] = s

training_data = list(map(lambda s, t: (list(s), t.split()), sent, tag))

print(training_data)

with open("./data/training_data/training_data.json", "w", encoding="utf8") as file:
    json.dump(training_data, file, ensure_ascii=False)