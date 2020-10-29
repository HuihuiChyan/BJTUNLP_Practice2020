import os
import csv
def read_test(file):
    path = './'+file+'.csv'
    f = open(path,'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Text','Label'])
    path_in = './CoNLL2003_NER/'+file+'/seq.in'
    path_out = './CoNLL2003_NER/'+file+'/seq.out'
    with open(path_in,encoding='utf-8') as file_input,open(path_out,encoding='utf-8') as file_output:
        for line1,line2 in zip(file_input,file_output):
            csv_writer.writerow([line1,line2])

read_test('train')
read_test('valid')
read_test('test')