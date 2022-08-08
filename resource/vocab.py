import json
result =list()
with open("./resource/vocab.txt", encoding="utf8") as file:
    data = file.read().splitlines()

vocab = {}
for value, key in enumerate(data):
    vocab[key] = value

with open("./resource/vocab.json", "w", encoding="utf8") as file:
    json.dump(vocab, file, ensure_ascii=False)