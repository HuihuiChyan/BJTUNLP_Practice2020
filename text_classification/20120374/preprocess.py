import os
import csv

def read_files(filetype):
    path = './aclImdb'
    file_list_pos,file_list_neg =[],[]

    pos_path = path + filetype + "/pos/"
    for f in os.listdir(pos_path):
        file_list_pos += [pos_path + f]

    neg_path = path + filetype + "/neg/"
    for f in os.listdir(neg_path):
        file_list_neg += [neg_path +f]

    path_csv = "./"+filetype+'.csv'
    f = open(path_csv,'w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Text','Label'])

    for file in file_list_pos:
        with open(file,encoding = 'utf8') as file_input:
            csv_writer.writerow([" ".join(file_input.readlines()),'1'])
            #all_texts.append(rm_tags(" ".join(file_input.readlines())))

    for file in file_list_neg:
        with open(file,encoding = 'utf8') as file_input:
            csv_writer.writerow([" ".join(file_input.readlines()),'0'])

    f.close()

#read_files("/train")
def read_test(test_path):
    f = open('./test.csv','w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Text','Label'])
    with open(test_path,encoding='utf-8') as file_input:
        for line in file_input:
            csv_writer.writerow([line,'-1'])

    f.close()

#read_test('./test.txt')



