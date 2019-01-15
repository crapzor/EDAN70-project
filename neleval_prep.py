"""
Converts the gold standard and the predicted files into formatting
neleval can process. 
"""

from docria.storage import DocumentIO
from docria.algorithm import dfs
import pickle


def read_sentences(file):
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


# saves into file in formatting for neleval
def tac16_conv(input_file, output_file):
    sentences = read_sentences(input_file)
    f_out = open(output_file, 'w')
    idx_st = 0
    idx_en = 0
    n = 0
    nill_nbr = 0
    k1 = 'LDC'
    k2 = 0
    k3 = ''
    k4 = 'TESTA'
    k5 = ''
    k6 = ''
    k7 = 'NAM'
    k8 = '1.0'
    kol = list()

    for sentence in sentences:
        mention = ''
        rows = sentence.split('\n')
        for row in rows:
            n += 1
            r = row.split(' ')
            if r[-2][0] == 'B' and mention:
                k1_l = k1
                k2_l = 'EDL16_EVAL_' + str(k2).zfill(5)
                k3_l = mention
                k4_l = 'ENG_NW_' + k4 + ':' + str(idx_st) + '-' + str(idx_en)
                k5_l = k5
                k6_l = k6
                k7_l = k7
                k8_l = k8
                kol = [k1_l, k2_l, k3_l, k4_l, k5_l, k6_l, k7_l, k8_l]
                tac16_row = '\t'.join(kol)
                f_out.write(tac16_row)
                f_out.write('\n')
                mention = ''

            if r[-2][0] == 'B' and not mention:
                mention = r[0].lower()
                k2 += 1
                if r[-1] == 'NILL': ## tänk till när ni ändrar nill
                    nill_nbr += 1
                    k5 = 'NIL' + str(nill_nbr).zfill(5)
                else:
                    k5 = r[-1]
                idx_st = n
                idx_en = n
                k6 = r[-2][2:]
            if r[-1][0] == 'I':
                mention += ' ' + r[0].lower()
                idx_en = n
            if r[-1][0] == 'O' and mention:
                k1_l = k1
                k2_l = 'EDL16_EVAL_' +str(k2).zfill(5)
                k3_l = mention
                k4_l = 'ENG_NW_' + k4 + ':' + str(idx_st) + '-' + str(idx_en)
                k5_l = k5
                k6_l = k6
                k7_l = k7
                k8_l = k8
                kol = [k1_l, k2_l, k3_l, k4_l, k5_l, k6_l, k7_l, k8_l]
                tac16_row = '\t'.join(kol)
                f_out.write(tac16_row)
                f_out.write('\n')
                mention = ''
        n += 1


if __name__ == '__main__':
    testa_gold_in = 'Gold_Standard.testa'
    testb_gold_in = 'Gold_Standard.testb'
    testa_pred_in = 'predicted_no_space.testa'
    testb_pred_in = 'predicted_no_space.testb'

    # make sure tac16 folder exists!
    testA_gold_out = 'tac16/TESTA_GOLD'
    testB_gold_out = 'tac16/TESTB_GOLD'
    testA_pred_out = 'tac16/TESTA_PRED'
    testB_pred_out = 'tac16/TESTB_PRED'

    tac16_conv(testa_gold_in, testA_gold_out)
    tac16_conv(testb_gold_in, testB_gold_out)
    tac16_conv(testa_pred_in, testA_pred_out)
    tac16_conv(testb_pred_in, testB_pred_out)
