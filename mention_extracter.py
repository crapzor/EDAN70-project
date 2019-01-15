"""
This:   * extracts information from docria
        * finds, and adds, the most frequently occuring wkd code for a mention
        to the file with the predictions.
"""

from docria.storage import DocumentIO
from docria.algorithm import dfs
import os
import pickle
import tkinter.filedialog


# returns sentences in file
def read_sentences(file):
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


# creates dictionary of all mentions in file's sentences
def file_mention_dict(sentences, mention_dict):
    for sentence in sentences:
        mention = ''
        rows = sentence.split('\n')
        for row in rows:
            r = row.split()
            if r[-1][0] == 'B' and mention:
                if mention not in mention_dict:
                    mention_dict[mention] = dict()
                mention = r[0].lower()
            if r[-1][0] == 'B' and not mention:
                mention = r[0].lower()
            if r[-1][0] == 'I':
                mention += ' ' + r[0].lower()
            if r[-1][0] == 'O' and mention:
                if mention not in mention_dict:
                    mention_dict[mention] = dict()


# retruns list of all files in directory fn
def files(fn):
    if os.path.exists(fn):  # if such path exist
        files = []
        for file_name in os.listdir(fn):  # for each file in the directory fn
            files.append(file_name)
        return files


# saves dictionary to file name in directory obj
def save_obj(dictionary, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


# loads dictionary from file name in directory obj
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# extract the different entities and their wkd codes from docria
def docria_extractor(files, docria_dir, mention_dict):
    for file in files:
        reader = DocumentIO.read(docria_dir + '/' + file)
        while True:
            temp = next(reader, None)
            # if next element is None break
            if temp == None:
                break
            # if text is in this iterator
            if temp.texts:
                for ent in temp.layers['anchor']:
                    if 'target_wkd' in ent:
                        entity = str(ent['text']).lower()
                        wkd = str(ent['target_wkd'])
                        if entity in mention_dict:
                            if wkd not in mention_dict[entity]:
                                mention_dict[entity][wkd] = 1
                            else:
                                mention_dict[entity][wkd] += 1
                        else:
                            mention_dict[entity] = dict()
                            mention_dict[entity][wkd] = 1


# extract the most frequent wkd code for an entity
def most_freq_wkd_dict(mention_dict, entity_dict):
    for ent in mention_dict.keys():
        if list(mention_dict[ent].values()):
            st = max(list(mention_dict[ent].values()))
            x = list(mention_dict[ent].values()).index(st)
            wkd = list(mention_dict[ent].keys())[x]
            entity_dict[ent] = 'Q' + wkd


# adds the wkd codes to the files
def add_to_files(sentences, entity_dict, file_name):
    f_out = open(file_name, 'w')
    for sentence in sentences:
        rows = sentence.split('\n')
        for i in range(len(rows)):
            r = rows[i].split()
            f_out.write(r[0] + ' ' + r[1] + ' ' + r[2] + ' ' + r[3])
            if r[-1][0] == 'B': # finds row chunked as B
                entry = r[0].lower()
                j = 1
                while (i + j) < (len(rows) - 1) and rows[i + j].split()[-1][0] is 'I':
                    entry += ' ' + rows[i + j].split()[0].lower() # all the consecutive words chunked as I is added to the mention
                    j += 1
                if entry in entity_dict:
                    f_out.write(' ' + entity_dict[entry]) # write the mention's wkd code
                else:
                    f_out.write(' NILL') # otherwise write NILL
            f_out.write('\n')
        f_out.write('\n')
    f_out.close()



if __name__ == '__main__':
    docria_dir = 'enwiki_/enwiki'
    testa_file = 'predicted_eng.testa'
    testb_file = 'predicted_eng.testb'
    sentencea = read_sentences(testa_file)
    sentenceb = read_sentences(testb_file)
    first_run = False

    if first_run:
        mention_dict = dict()
        entity_dict = dict()

        file_mention_dict(sentencea, mention_dict)
        file_mention_dict(sentenceb, mention_dict)
        docria_files = files(docria_dir)
        docria_extractor(docria_files, docria_dir, mention_dict)
        most_freq_wkd_dict(mention_dict, entity_dict)
        # before saving, make sure folder obj exists!
        save_obj(mention_dict, 'mentions')
        save_obj(entity_dict, 'entity_dict')
        exit()
    else:
        entity_dict = load_obj('entity_dict')

    add_to_files(sentencea, entity_dict, 'predicted_wkd_eng.testa')
    add_to_files(sentenceb, entity_dict, 'predicted_wkd_eng.testb')
