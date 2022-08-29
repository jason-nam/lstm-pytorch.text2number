import json
import os

current_path = os.path.dirname(os.path.abspath(__file__))
def main():
    with open(os.path.join(current_path,"td_sent_filter.json"),"r",encoding ="UTF-8-sig") as f:
        sentences = json.load(f)
    with open(os.path.join(current_path,"td_tag_filter.json"),"r",encoding ="UTF-8-sig") as f:
        BIO = json.load(f)
        
    with open("train.tsv", 'w', encoding='utf-8') as f:
        for ind, sentence in enumerate(sentences[:(len(sentences)//10)*9]):
            f.write(sentence + "@@@" + BIO[ind] + "\n")
    
    with open("test.tsv", 'w', encoding='utf-8') as f:
        list_sentences = sentences[(len(sentences)//10)*9:]
        list_BIO = BIO[(len(sentences)//10)*9:]
        for ind, sentence in enumerate(list_sentences):
            f.write(sentence + "@@@" + list_BIO[ind] + "\n")
            
if __name__ == "__main__":
    main()