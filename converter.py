"""
IOBv1 to IOBv2 converter
Takes CoNLL file in IOBv1 formatting and converts
it to IOBv2 formatting.
"""

def read_sentences(file):
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def IOBv1_to_v2(file_in, file_name):
    sentences = read_sentences(file_in)
    f_out = open(file_name, 'w')
    for sentence in sentences:
        rows = sentence.split('\n')
        trigger = 'O'
        for row in rows:
            r = row.split()
            f_out.write(r[0] + ' ' + r[1] + ' ')
            if r[3][0] == 'I' and trigger == 'O':
                trigger = 'B'
            else:
                trigger = r[3][0]

            f_out.write(trigger + r[3][1:])
            f_out.write('\n')

        f_out.write('\n')
    f_out.close()


if __name__ == '__main__':
    IOBv1_to_v2('corpus/eng.testa', 'corpus/conv_eng.testa')
    IOBv1_to_v2('corpus/eng.testb', 'corpus/conv_eng.testb')
    IOBv1_to_v2('corpus/eng.train', 'corpus/conv_eng.train')
