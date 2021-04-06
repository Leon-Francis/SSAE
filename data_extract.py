if __name__ == '__main__':
    data_file_path = './calc_BertScore_ppl_SNLI_Bert_without_adv.log'
    output_path = './data/' + data_file_path[2:-4] + '.txt'
    with open(data_file_path, 'r',
              encoding='utf-8') as r_f, open(output_path,
                                             'w',
                                             encoding='utf-8') as w_f:
        out_list = [['0' for i in range(3)] for j in range(6)]
        i = 0
        for line in r_f:
            line = line.strip('\n')
            if line == '':
                continue
            line_list = line.split('=')
            if len(line_list) <= 1:
                continue
            if line_list[0] == 'attack_acc':
                out_list[i][2] = line_list[1]
                continue
            if line_list[0] == 'ppl':
                out_list[i][1] = line_list[1]
                continue
            if line_list[0] == 'bert_score':
                out_list[i][0] = line_list[1]
                i += 1
                continue
            if i == 6:
                if line_list[0] == 'gen_time':
                    for t in range(6):
                        w_f.write('\t'.join(out_list[t]) + '\n\n')
                    w_f.write(line_list[1] + '\n\n')
                    i = 0
                    continue
