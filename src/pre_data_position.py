import random
import json
import copy
import re
from src import config

PAD_token = 0

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
        if config.USE_APE_word:
            if len(key)>1:
                word_lst.append(key)
        if config.USE_APE_char:
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

def merge_data(fold, fold_size, comb_index_data, pairs, index_len):

    test_start = fold_size * fold
    test_end = fold_size * (fold + 1)

     # 构造正样本
    temp_pairs = []
    temp_pairs_pos = []
    temp_pairs_neg = []
    
    # 测试样本的index
    test_index_list = range(test_start, test_end)

     # 保存样本对应的正样本
    sample_pos = {}

    for pair in comb_index_data:
        # 这个i是在原始样本pair中的index
        i = pair[0]
        j = pair[1]
        
        # 筛选掉在测试集中的样本
        if i in test_index_list or j in test_index_list:
            continue

        sample = pairs[i]
        pos_sample = pairs[j]
        
        if i not in sample_pos:
            sample_pos[i] = []
        sample_pos[i].append(j)

        temp_pairs.append(sample)
        temp_pairs_pos.append(pos_sample)

    # 构造负样本 index_len 为样本总数
     
    for pair in comb_index_data:
        neg_index = numpy.random.randint(0, high=index_len)
        i = pair[0]
        j = pair[1]
        
         # 筛选掉在测试集中的样本
        if i in test_index_list or j in test_index_list:
            continue

        pos_index_list = sample_pos[i]

        while neg_index in pos_index_list or neg_index in test_index_list or neg_index == i :
            neg_index = numpy.random.randint(0, high=index_len)

        temp_pairs_neg.append(pairs[neg_index]) 

     # 把所有的数据集拼接到一起
    pairs_len = len(temp_pairs)
    # 合并三个样本
    pairs_trained = temp_pairs + temp_pairs_pos + temp_pairs_neg

    return pairs_trained, pairs_len

class Lang:
# Lang()

    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence, output = False):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                if output == True:
                   print("Add_sen_to_vocab:{}".format(word))
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)
        print("In build lang and the number of op is:{}".format(self.num_start))

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


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


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = {}
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
    return out_data

# for testing equation
# def out_equation(test, num_list):
#     test_str = ""
#     for c in test:
#         if c[0] == "N":
#             x = num_list[int(c[1:])]
#             if x[-1] == "%":
#                 test_str += "(" + x[:-1] + "/100.0" + ")"
#             else:
#                 test_str += x
#         elif c == "^":
#             test_str += "**"
#         elif c == "[":
#             test_str += "("
#         elif c == "]":
#             test_str += ")"
#         else:
#             test_str += c
#     return test_str


def transfer_num_comp(data):  
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
    for d in data:
        nums = []
        input_seq = ['[CLS]']
        if config.USE_APE:
            seg = d[0].strip().split(" ")
            equations = d[1]
        else:
            seg = d["segmented_text"].strip().split(" ")
            equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.extend(list(s))
        input_seq.append('[SEP]')
        if copy_nums < len(nums):
           copy_nums = len(nums)
        if len(nums)>max_num_size:
           max_num_size = len(nums)
        
        nums_sort = sorted(nums, key=lambda x:  float(x[:-1])*0.01  if re.search("\d+\.\d+%|\d+%", x) else  (float(x[0: x.index('(') ]) * float(x.split("/")[0][( x.index('(') +1):])) /  float(x.split("/")[1][0:-1]) if  re.search("\d*\(\d+/\d+\)\d*", x) and x[0]!='(' else    float(x[1:-1].split("/")[0]) /  float(x[1:-1].split("/")[1])    if re.search("\d*\(\d+/\d+\)\d*", x) else float(x))#, reverse=True) 
        nums_gravity = {}
        for i, num in enumerate(nums_sort):
            nums_gravity[num] = i

        num_pos = []
        index = 0
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
               
        assert len(nums) == len(num_pos)
        if len(input_seq) > config.Max_Question_len and config.USE_APE :
            count_too_lang+=1
            continue



        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            #     if num[0]!='(':
            #        print("num:{}".format(num))
            #        index = num.index('(')
            #        print("结果:{}".format(float(num[0:index]) * float(num[(index+1):-1].split("/")[0]) /  float(num[(index+1):-1].split("/")[1])) )
            #     else:
            #        print("结果:{}".format(float(num[1:-1].split("/")[0]) /  float(num[1:-1].split("/")[1])) )

            # if  re.search("\d+\.\d+%|\d+%", num):
            #     print("num:{} \n float(x[:-1])*0.01:{}".format(num, float(num[:-1])*0.01))

        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        # for x in nums:
        #     if re.search("\d+\.\d+%|\d+%", x):
        #         print("x[:-1])*0.01  :{}".format(float(x[:-1])*0.01 ))
        #     else:
        #         if  re.search("\d*\(\d+/\d+\)\d*", x) and x[0]!='(':
        #            print("x:{}".format(x))
        #            print("x[0: num.index('('):{}".format(x[0: x.index('(')] ))
        #            print("x.split(""):{}".format(x.split("/")[0][   ( x.index('(')+1):        ]                  ))
        #            print("float(x.split("")[0][( num.index('(') +1):-1]):{}".format(x.split("/")[0][( x.index('(') +1):-1]))
        #            print("float(x.split("")[1][( num.index('(') +1):-1]):{}".format(float(x.split("/")[1][0:-1])  ))

        #            print("(float(x[0: num.index('(') ]) * float(x[( num.index('(') +1):-1].split("")[0])) /  float(x[( num.index('(') +1):-1].split("")[1]):{}".format(  ( float(x[0: x.index('(') ]) * float(x.split("/")[0][( x.index('(') +1):]) ) /  float(x.split("/")[1][0:-1])       ))
        #         else:
        #            if re.search("\d*\(\d+/\d+\)\d*", x):
        #               print("float(x[1:-1].split("")[0]) /  float(x[1:-1].split("")[1]):{}".format(float(x[1:-1].split("/")[0]) /  float(x[1:-1].split("/")[1])))
        #            else:
        #               print("float(x):{}".format(float(x)))

        
        #print("nums:{}".format(nums))
        
            

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        
        #nums_gravity_list = []
       
        
        if  config.USE_APE:
            if len(out_seq) == 0 or len(out_seq)> config.Max_Expression_len:
                exp_too_lang+=1
            else:
                pairs.append((input_seq, out_seq, nums, num_pos))
        else:
            pairs.append((input_seq, out_seq, nums, num_pos))
    
    if config.USE_APE:
        print("data_set_size is %d, num of exp>60  is %d,about %.4f" %(len(pairs),exp_too_lang,float(exp_too_lang)/len(pairs)))
        print("data_set_size is %d, num of problem>150  is %d,about %.4f" %(len(pairs),count_too_lang,float(count_too_lang)/len(pairs)))

    temp_g = []
    if config.USE_APE:
        orderList=list(generate_nums_dict.values())  
        orderList.sort(reverse=True)  
        max_order_list=orderList[0:10]
        min_generate_vocab_appear=max_order_list[-1]
        for g in generate_nums:
            if generate_nums_dict[g] >= min_generate_vocab_appear:
                temp_g.append(g)
        print("generate_num size is %d" %(len(temp_g)))
        print("min_generate_vocab_appear times is %d" %(min_generate_vocab_appear))
    else:
        print("generate_nums before:{}".format(generate_nums))
        for g in generate_nums:
            if generate_nums_dict[g] >= 5:
                temp_g.append(g)
        print("generate_nums after:{}".format(temp_g))
    
    print("max_num_size:{}".format(max_num_size))
    return pairs, temp_g, copy_nums

def transfer_num(data): 
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
    for d in data:
        nums = []
        input_seq = ['[CLS]']
        if config.USE_APE:
            seg = d[0].strip().split(" ")
            equations = d[1]
        else:
            seg = d["segmented_text"].strip().split(" ")
            equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.extend(list(s))
        input_seq.append('[SEP]')
        if copy_nums < len(nums):
           copy_nums = len(nums)
        if len(nums)>max_num_size:
           max_num_size = len(nums)
        
        nums_sort = sorted(nums, key=lambda x:  float(x[:-1])*0.01  if re.search("\d+\.\d+%|\d+%", x) else  (float(x[0: x.index('(') ]) * float(x.split("/")[0][( x.index('(') +1):])) /  float(x.split("/")[1][0:-1]) if  re.search("\d*\(\d+/\d+\)\d*", x) and x[0]!='(' else    float(x[1:-1].split("/")[0]) /  float(x[1:-1].split("/")[1])    if re.search("\d*\(\d+/\d+\)\d*", x) else float(x))#, reverse=True) 
        nums_gravity = {}
        for i, num in enumerate(nums_sort):
            nums_gravity[num] = i

        num_pos = []
        index = 0
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
                #print("here")
                input_seq[i] = j + str(nums_gravity[nums[index]])
                #nums_gravity_list.append(nums_gravity[nums[index]])
                index = index + 1
        assert len(nums) == len(num_pos)
        if len(input_seq) > config.Max_Question_len and config.USE_APE :
            count_too_lang+=1
            continue



        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
            #     if num[0]!='(':
            #        print("num:{}".format(num))
            #        index = num.index('(')
            #        print("结果:{}".format(float(num[0:index]) * float(num[(index+1):-1].split("/")[0]) /  float(num[(index+1):-1].split("/")[1])) )
            #     else:
            #        print("结果:{}".format(float(num[1:-1].split("/")[0]) /  float(num[1:-1].split("/")[1])) )

            # if  re.search("\d+\.\d+%|\d+%", num):
            #     print("num:{} \n float(x[:-1])*0.01:{}".format(num, float(num[:-1])*0.01))

        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        # for x in nums:
        #     if re.search("\d+\.\d+%|\d+%", x):
        #         print("x[:-1])*0.01  :{}".format(float(x[:-1])*0.01 ))
        #     else:
        #         if  re.search("\d*\(\d+/\d+\)\d*", x) and x[0]!='(':
        #            print("x:{}".format(x))
        #            print("x[0: num.index('('):{}".format(x[0: x.index('(')] ))
        #            print("x.split(""):{}".format(x.split("/")[0][   ( x.index('(')+1):        ]                  ))
        #            print("float(x.split("")[0][( num.index('(') +1):-1]):{}".format(x.split("/")[0][( x.index('(') +1):-1]))
        #            print("float(x.split("")[1][( num.index('(') +1):-1]):{}".format(float(x.split("/")[1][0:-1])  ))

        #            print("(float(x[0: num.index('(') ]) * float(x[( num.index('(') +1):-1].split("")[0])) /  float(x[( num.index('(') +1):-1].split("")[1]):{}".format(  ( float(x[0: x.index('(') ]) * float(x.split("/")[0][( x.index('(') +1):]) ) /  float(x.split("/")[1][0:-1])       ))
        #         else:
        #            if re.search("\d*\(\d+/\d+\)\d*", x):
        #               print("float(x[1:-1].split("")[0]) /  float(x[1:-1].split("")[1]):{}".format(float(x[1:-1].split("/")[0]) /  float(x[1:-1].split("/")[1])))
        #            else:
        #               print("float(x):{}".format(float(x)))

        
        #print("nums:{}".format(nums))
        
            

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        
        #nums_gravity_list = []
       
        
        if  config.USE_APE:
            if len(out_seq) == 0 or len(out_seq)> config.Max_Expression_len:
                exp_too_lang+=1
            else:
                pairs.append((input_seq, out_seq, nums, num_pos))
        else:
            pairs.append((input_seq, out_seq, nums, num_pos))
    
    if config.USE_APE:
        print("data_set_size is %d, num of exp>60  is %d,about %.4f" %(len(pairs),exp_too_lang,float(exp_too_lang)/len(pairs)))
        print("data_set_size is %d, num of problem>150  is %d,about %.4f" %(len(pairs),count_too_lang,float(count_too_lang)/len(pairs)))

    temp_g = []
    if config.USE_APE:
        orderList=list(generate_nums_dict.values())  
        orderList.sort(reverse=True)  
        max_order_list=orderList[0:10]
        min_generate_vocab_appear=max_order_list[-1]
        for g in generate_nums:
            if generate_nums_dict[g] >= min_generate_vocab_appear:
                temp_g.append(g)
        print("generate_num size is %d" %(len(temp_g)))
        print("min_generate_vocab_appear times is %d" %(min_generate_vocab_appear))
    else:
        print("generate_nums before:{}".format(generate_nums))
        for g in generate_nums:
            if generate_nums_dict[g] >= 5:
                temp_g.append(g)
        print("generate_nums after:{}".format(temp_g))
    
    print("max_num_size:{}".format(max_num_size))
    return pairs, temp_g, copy_nums


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((input_seq, eq_segs, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def transfer_roth_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = {}
    generate_nums = {}
    copy_nums = 0
    for key in data:
        d = data[key]
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs[key] = (input_seq, eq_segs, nums, num_pos)

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res

def indeces_from_sentence_via_tokenizer(tokenizer, sentence):
    return  tokenizer.convert_tokens_to_ids(sentence)

def prepare_data(tokenizer, pairs_trained, pairs_tested_list, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            # print("pair[1]:{}".format(pair[1]))
            output_lang.add_sen_to_vocab(pair[1] , output = True)
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indeces_from_sentence_via_tokenizer(tokenizer, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack#,
                            #pair[4]   #gravity
                            ))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    test_pairs_list = []
    for pairs_tested in pairs_tested_list:
        test_pairs = []
        for pair in pairs_tested:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            #print("pair[0]:{}".format(pair[0]))
            input_cell = indeces_from_sentence_via_tokenizer(tokenizer, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            #print("input_cell:{}".format(input_cell))
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack#,
                            #pair[4]   #gravity
                            ))
        test_pairs_list.append(test_pairs)
        print('Number of testing data %d' % (len(test_pairs)))
    
    return input_lang, output_lang, train_pairs, test_pairs_list

def prepare_data_5fold(tokenizer, pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            # print("pair[1]:{}".format(pair[1]))
            output_lang.add_sen_to_vocab(pair[1] , output = True)
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indeces_from_sentence_via_tokenizer(tokenizer, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack#,
                            #pair[4]   #gravity
                            ))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    
    test_pairs = []
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        #print("pair[0]:{}".format(pair[0]))
        input_cell = indeces_from_sentence_via_tokenizer(tokenizer, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        #print("input_cell:{}".format(input_cell))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                        pair[2], pair[3], num_stack#,
                        #pair[4]   #gravity
                        ))
   
    print('Number of testing data %d' % (len(test_pairs)))
    
    return input_lang, output_lang, train_pairs, test_pairs


def get_single_example_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = input_length
    sentence_length = input_length
    quantity_cell_list = group
    num_list = num_value
    id_num_list = num_pos
    graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_attbet = get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    graph_lower = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    #graph_newc1 = get_quantity_graph1(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
    batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)
    return batch_graph

def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        # train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack, pair[4]])
        train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack))
    print('Number of testind data %d' % (len(test_pairs)))
    # the following is to test out_equation
    # counter = 0
    # for pdx, p in enumerate(train_pairs):
    #     temp_out = allocation(p[2], 0.8)
    #     x = out_equation(p[2], p[4])
    #     y = out_equation(temp_out, p[4])
    #     if x != y:
    #         counter += 1
    #     ans = p[7]
    #     if ans[-1] == '%':
    #         ans = ans[:-1] + "/100"
    #     if "(" in ans:
    #         for idx, i in enumerate(ans):
    #             if i != "(":
    #                 continue
    #             else:
    #                 break
    #         ans = ans[:idx] + "+" + ans[idx:]
    #     try:
    #         if abs(eval(y + "-(" + x + ")")) < 1e-4:
    #             z = 1
    #         else:
    #             print(pdx, x, p[2], y, temp_out, eval(x), eval("(" + ans + ")"))
    #     except:
    #         print(pdx, x, p[2], y, temp_out, p[7])
    # print(counter)
    return input_lang, output_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []

    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []

    

    # num_gravity_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []

        #num_gravity_bacth = []
        
        for i, li, j, lj, num, num_pos, num_stack in batch:   #, num_gravity
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))

            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))

            #num_gravity_bacth.append(num_gravity)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        #num_gravity_batches.append(num_gravity_bacth)
       
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches#, num_gravity_batches


def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex


