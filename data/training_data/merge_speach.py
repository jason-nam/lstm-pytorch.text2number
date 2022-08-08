with open("./data/training_data/original.txt", encoding="utf-8") as file:
    sent = file.read().splitlines()

with open("./data/training_data/bio.txt") as file:
    tag = file.read().splitlines()

training_data = list(map(lambda s, t: (list(s), t.split()), sent, tag))
