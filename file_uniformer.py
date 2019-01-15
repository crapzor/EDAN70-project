"""
Fixes the incorrect wkd codes and adds them to the gold standard files.
Also removes the blank lines in the gold standard, and the predicted, files. 
"""

import pickle


# extracts the test A and test B files from the file recieved by AIDA
# AIDA returns one file containing train, test A and test B
def split_aida(input_file, output_file):
    output_testa = output_file + '.testa'
    output_testb = output_file + '.testb'
    f = open(input_file).read().strip()
    f_outa = open(output_testa, 'w')
    f_outb = open(output_testb, 'w')
    words = f.split('\n')
    triggera = False
    triggerb = False
    for i in range(len(words)):
        if not triggera and not triggerb and 'DOCSTART' in words[i] and 'testa' in words[i]:
            triggera = True
        if triggera and not triggerb and 'DOCSTART' in words[i] and 'testb' in words[i]:
            triggerb = True
        if triggera and not triggerb:
            f_outa.write(words[i])
            if 'DOCSTART' in words[i] and i > 1:
                f_outa.write('\n\t\t\t\t\t\t')
            f_outa.write('\n')
        if triggera and triggerb:
            f_outb.write(words[i])
            if 'DOCSTART' in words[i] and i > 1:
                f_outb.write('\n\t\t\t\t\t\t')
            f_outb.write('\n')
    f_outa.close()
    f_outb.close()


def read_sentences(file):
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


# changes the incorrect wkd codes in the gold standard file
# based on the freebase codes and the freebase to wkd conversion dictionary
def fix_wkd(file, fb2wkd_dict, file_name):
    sentences = read_sentences(file)
    f = open(file_name, 'w')
    for sentence in sentences:
        rows = sentence.split('\n')
        for row in rows:
            #idx+=1
            r = row.split('\t')
            if r[-1] and len(r[-1]) > 3:
                r[-1] = r[-1][1] + '.' + r[-1][3:]
                if r[-1] in fb2wkd_dict:
                    r[-2] = str(fb2wkd_dict[r[-1]])
            line = '\t'.join(r)
            line += '\n'
            f.write(line)
    f.close()


# returns the contents of the file as a list
# for different formatting: both aida and conll
def file2list(filename, type):
    outlist = list()
    f = open(filename).read().strip()
    sentences = f.split('\n\n')
    if type == 'conll':
        for sentence in sentences:
            rows = sentence.split('\n')
            for row in rows:
                if 'DOCSTART' not in row:
                    outlist.append(row)
    elif type == 'aida':
        for sentence in sentences:
            rows = sentence.split('\n')
            for row in rows:
                r = row.split('\t')
                if r[0] and 'DOCSTART' not in r[0]:
                    outlist.append(row)
    return outlist


# removes all blank rows from file_in and saves it in file_out
def change_predicted(file_in, file_out):
    f = open(file_out, 'w')
    words = file2list(file_in, 'conll')
    for i in range(len(words)):
        word = words[i].split(' ')
        w = list()
        for j in range(len(word)):
            if j != 2:
                w.append(word[j])
        line = ' '.join(w)
        line += '\n'
        f.write(line)
    f.close()


# adds the wkd codes from aida to the conll gold standard file
# and saves it to filename
def file2file(aida, gold, filename):
    f = open(filename, 'w')
    words = file2list(gold, 'conll')
    worda = file2list(aida, 'aida')
    for i in range(len(words)):
        word_g = words[i].split('\t')
        word_a = worda[i].split('\t')
        line = ' '.join(word_g)
        if len(word_a) > 4:
            if word_a[1] == 'B':
                if word_a[-2]:
                    line += ' ' + 'Q' + word_a[-2]
                else:
                    line += ' ' + 'NILL'
            line += '\n'
        f.write(line)
    f.close()


# loads dictionary
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    split_aida('aida.csv', 'aida') # extracts test a and b from the aida file

    aida_a = 'aida.testa'
    aida_b = 'aida.testb'
    basefilea = 'corpus/conv_eng.testa'
    basefileb = 'corpus/conv_eng.testb'
    gold_entity_dict = dict()

    wkd2fb = load_obj('wkd2fb') # dictionary recieved by Marcus
    fb2wkd = {v: k for k, v in wkd2fb.items()} # invert dictionary

    fix_wkd(aida_a, fb2wkd, 'aida_proper.testa') # fix wkd codes in aida
    fix_wkd(aida_b, fb2wkd, 'aida_proper.testb')

    file2file('aida_proper.testa', basefilea, 'Gold_Standard.testa') # join aida and conll and remove blank rows
    file2file('aida_proper.testb', basefileb, 'Gold_Standard.testb')

    change_predicted('predicted_wkd_eng.testa', 'predicted_no_space.testa') # remove blank rows from predicted files
    change_predicted('predicted_wkd_eng.testb', 'predicted_no_space.testb')
