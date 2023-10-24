# # coding: utf-8
# from src.train_and_evaluate import *
# from src.models import *
# import time
# import torch.optim

# from src.expressions_transfer import *

# from transformers import AutoTokenizer
# from tqdm import tqdm

# from src import config
# from src.schedule_sampling import * 
import json
import re

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def get_train_test_fold(ori_path, prefix, data, pairs):
    """
        ori_path:  './data/'
        prefix:    '23k_processed.json'
        data:      加载的原始数据
        pairs:     经过transfer_num处理后的数据
    """
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []


    for item, pair in zip(data, pairs):
        pair = list(pair)
        # pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold



# batch_size = 16
# embedding_size = config.embedding_size#128
# hidden_size = config.hidden_size

# learning_rate = 5e-5
# weight_decay = 1e-5
# beam_size = 5
# n_layers = 2

# # 在这里弄个tokenizer就行
# tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
# tokenizer.add_special_tokens({'additional_special_tokens':['NUM']})
# vocab_size = len(tokenizer)
def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def load_data(filename,divide=1):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    print(filename)
    question_num=0
    not_equal=0
    cannot_eval=0
    word_lst=[]
    word_space=[]
    word_dense=[]
    for line in open('data/apewordVocab.txt'):
        key=line.strip().split()[0]
        if USE_APE_word:
            if len(key)>1:
                word_lst.append(key)
        if USE_APE_char:
            if len(key)>5:
                word_lst.append(key)
    word_lst.sort(key = lambda i:len(i),reverse=True) 
    for key in word_lst:
        value_list=[]
        for char in range(0,len(key),1):
            value_list.append(key[char:char+1])
        
        value=" ".join(value_list)

        word_space.append(" "+value+" ")
        word_dense.append(" "+key+" ")

    for l in open(filename):
        question_num+=1
        #if question_num%50000==0:
        #    print(question_num)
        if question_num%divide==0:
            l = json.loads(l)
            question, equation, answer = l['segmented_text'].strip(), l['equation'], l['ans']
            flag=0
            #if "(".encode("UTF-8") in question or " / ".encode("UTF-8") in question or "%".encode("UTF-8") in question:
            #    print(question)
            #    print(equation)s
            #    print(answer)
            #    flag=1
            # 处理带分数
            question = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', question)
            equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
            equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
            #equation = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', equation)
            #answer = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', answer)
            equation = re.sub('(\d+) \(', '\\1(', equation)
            answer = re.sub('(\d+) \(', '\\1(', answer)
            # 分数去括号
            #question = re.sub('\((\d+/\d+)\)', '\\1', question)
            # 分数合并
            question = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', question)
            equation = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', equation)
            
            # 分数加括号
            #question = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', question)
            #equation = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', equation)
            # 处理百分数
            question = re.sub('([\.\d]+)%', '(\\1/100)', question)
            equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
            answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
            # 冒号转除号、剩余百分号处理
            question = question.replace('%', ' / 100')
            equation = equation.replace(':', '/').replace('%', '/100')
            answer = answer.replace(':', '/').replace('%', '/100')
            equation = equation.replace('"千米/小时"', '')
            #if flag==1:
            #    print(question)
            #    print(equation)
            #    print(answer)
            #    flag=1
            if equation[:2] == 'x=':
                equation = equation[2:]

            idx_ = 0
            question=" "+question+" "
            for idx_ in range(len(word_dense)):
                if word_space[idx_] in question:
                    question=question.replace(word_space[idx_],word_dense[idx_])
            question=question.strip()
            try:
                if is_equal(eval(equation), eval(answer)):
                    D.append((question, equation, answer))
                else:
                    #print("not equal")
                    #print(question)
                    #print(equation)
                    #print(eval(equation))
                    #print(answer)
                    #print(eval(answer))
                    not_equal+=1
            except:
                #print(question)
                #print(equation)
                D.append((question, equation, answer))
                cannot_eval+=1
                continue
    with open(filename+"clear",'w') as wf2:
        for item in D:
            wf2.write(item[0]+"\n")
            wf2.write(item[1]+"\n")
    print(question_num)
    print(not_equal)
    print(cannot_eval)
    print(len(D))
    return D

data = load_data("train.ape.json")
print("len of data:{}".format(len(data)))






def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

def is_float(s):

    return sum([n.isdigit() for n in s.strip().split('.')]) >= 2

def is_fraction(s):

    return sum([n.isdigit() for n in s.strip().split('/')]) >= 2

def load_QA_data(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    count_too_lang=0
    exp_too_lang=0

    # 计算一条文本中最多有多少个数字
    max_num_size = 0
    pairs = []
    
    for d in data:
        data_dict ={}
        seg = d["original_text"].strip().split(" ")
        
        # 过滤分数
        seg_temp = []
        for index, element in enumerate(seg):
            if index >0 and index<(len(seg)-1) and element == "/" and seg[index-1].isdigit() and seg[index+1].isdigit():
                # print("len(seg_temp):{}".format(len(seg_temp)))
                # print("len(seg):{}".format(len(seg)))
                # print("index+1:{}".format(index+1))
                seg_temp[-1] = seg[index-1]+"/"+seg[index+1]
                continue

            if index>1 and seg[index].isdigit() and seg[index-1] == "/" and seg[index-2].isdigit():
                continue
            seg_temp.append(element)
        
        seg = seg_temp
        seg_temp = []
        for index, element in enumerate(seg):
            if index >0  and element == "%" and (seg[index-1].isdigit() or is_float(seg[index-1]) or is_fraction(seg[index-1])):
                seg_temp[-1] = seg[index-1]+"%"
                continue

            seg_temp.append(element)

        data_dict["original_text"] = " ".join(seg_temp)

        equations = d["equation"]
        data_dict["equation"] = equations

        ans = d["ans"]
        data_dict["ans"] = ans

        pairs.append(data_dict)
    #print("count:{}".format(count))
    return pairs


data_ = load_QA_data(data)

filename = "MathQA_train_process.json"
with open(filename, 'w') as file_obj:
  json.dump(data_, file_obj)



    

