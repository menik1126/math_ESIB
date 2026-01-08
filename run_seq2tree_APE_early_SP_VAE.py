# coding: utf-8
from src import config

from src.models_vae_divide import *
from src.train_and_evaluate_divide_vae import *


# from src.train_and_evaluate_prune import *
# from src.models_prune import *
import time
import torch.optim
import argparse

from src.expressions_transfer import *
from tqdm import tqdm


import torch.nn.utils.prune as prune
import pytorch_warmup as warmup
import os


def parse_args():
    parser = argparse.ArgumentParser(description='APE210K Full Model with Early Stopping')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden layer dimension')
    parser.add_argument('--embedding_size', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--model_name', type=str, default='roberta', help='Pretrained model name')
    parser.add_argument('--use_ape', type=str, default='true', help='Use APE dataset (true/false)')
    parser.add_argument('--warmup_period', type=int, default=3000, help='Warmup steps')
    parser.add_argument('--contra_weight', type=float, default=0.005, help='Contrastive loss weight')
    parser.add_argument('--test_interval', type=int, default=5, help='Test interval (epochs)')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Override config with command line arguments
batch_size = args.batch_size
hidden_size = args.hidden_size
learning_rate = args.learning_rate
embedding_size = args.embedding_size
n_epochs = args.n_epochs
beam_size = args.beam_size
model_name = args.model_name
warmup_period = args.warmup_period
contra_weight = args.contra_weight
test_interval = args.test_interval

# Update config module
config.hidden_size = hidden_size
config.embedding_size = embedding_size
config.n_epochs = n_epochs
config.MODEL_NAME = model_name
config.warmup_period = warmup_period
config.contra_weight = contra_weight
config.dropout = args.dropout
config.test_interval = test_interval

weight_decay = 1e-5
n_layers = 2

num_list_text = []
for d in range(config.quantity_num_ape):
    num_list_text.append('NUM'+str(d))


if model_name=='roberta':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif model_name=='roberta-large':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta_large/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif model_name =='xml-roberta':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif model_name =='xml-roberta-base':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif model_name =='bert-base-chinese':
    from transformers import AutoTokenizer
    print("model name:{}".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)


vocab_size = len(tokenizer)
dataset = "APE"


valid_data = load_data('data/ape/valid.ape.json',1)
print(valid_data[0])
print(valid_data[1])
train_data = load_data('data/ape/train.ape.json',1)
test_data = load_data('data/ape/test.ape.json',1)

# train_dataset
pairs, generate_nums, copy_nums = transfer_num(train_data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

# valid_dataset
pairs_from_test, _, _ = transfer_num(valid_data)
temp_pairs = []
for p in pairs_from_test:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

# test_dataset
pairs_from_valid, _, _ = transfer_num(test_data)
temp_pairs = []
for p in pairs_from_valid:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_validated = temp_pairs


input_lang, output_lang, train_pairs, (valid_pairs, test_pairs) = prepare_data(tokenizer, pairs_trained, [pairs_validated, pairs_tested], 5, generate_nums,
                                                            copy_nums, tree=True)



print("##############################")
print("input_lang words"+str(input_lang.n_words))
print("output_lang words"+str(output_lang.n_words))
print("generate nums:")
print(generate_nums)
print("copy number max nums"+str(copy_nums))
print("dataset_size:")
print(len(pairs))
print(len(pairs_from_test))
print(len(pairs_from_valid))
print("dataset_after indexed size:")
print(len(train_pairs))
print(len(test_pairs))
print(len(valid_pairs))

def indexes_to_sentence(lang, index_list, tree=False):
    res = []
    for index in index_list:
        if index < lang.n_words:
            res.append(lang.index2word[index])
    return res

UNK= output_lang.word2index["UNK"]
temp_pairs = []
i=0
for p in train_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
    else:
        i+=1
        if i<5:
            #print( " ".join(indexes_to_sentence(input_lang,p[0])))
            print( " ".join(indexes_to_sentence(output_lang,p[2])))

train_pairs=temp_pairs
temp_pairs = []
for p in test_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
test_pairs=temp_pairs
temp_pairs = []
for p in valid_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
valid_pairs=temp_pairs

print("##############################")
print("dataset_after erase UNK data:")
print(len(train_pairs))
print(len(test_pairs))
print(len(valid_pairs))


# Initialize models,here op_nums [PAD, +,- ,*,^,/]
encoder = EncoderSeq(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)


encoder_1 = EncoderSeq(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                n_layers=n_layers)
predict_1 = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                input_size=len(generate_nums))
generate_1 = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    embedding_size=embedding_size)
merge_1 = Merge(hidden_size=hidden_size, embedding_size=embedding_size)



# the embedding layer is  only for generated number embeddings, operators, and paddings
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

encoder_optimizer1 = torch.optim.AdamW(encoder_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
predict_optimizer1 = torch.optim.AdamW(predict_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
generate_optimizer1 = torch.optim.AdamW(generate_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
merge_optimizer1 = torch.optim.AdamW(merge_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)



encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[n_epochs//3], gamma=0.1)
predict_scheduler = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer, milestones=[n_epochs//3], gamma=0.1)
generate_scheduler = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer, milestones=[n_epochs//3], gamma=0.1)
merge_scheduler = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer, milestones=[n_epochs//3], gamma=0.1)

encoder_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer1, milestones=[n_epochs//3], gamma=0.1)
predict_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer1, milestones=[n_epochs//3], gamma=0.1)
generate_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer1, milestones=[n_epochs//3], gamma=0.1)
merge_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer1, milestones=[n_epochs//3], gamma=0.1)

if config.warm_up_stratege == "original":
    encoder_warmup_scheduler = warmup.UntunedLinearWarmup(encoder_optimizer)
    encoder_warmup_scheduler.last_step = -1 # initialize the step counter

    predict_warmup_scheduler = warmup.UntunedLinearWarmup(predict_optimizer)
    predict_warmup_scheduler.last_step = -1

    generate_warmup_scheduler = warmup.UntunedLinearWarmup(generate_optimizer)
    generate_warmup_scheduler.last_step = -1

    merge_warmup_scheduler = warmup.UntunedLinearWarmup(merge_optimizer)
    merge_warmup_scheduler.last_step = -1

    encoder_warmup_scheduler1 = warmup.UntunedLinearWarmup(encoder_optimizer1)
    encoder_warmup_scheduler1.last_step = -1 # initialize the step counter

    predict_warmup_scheduler1 = warmup.UntunedLinearWarmup(predict_optimizer1)
    predict_warmup_scheduler1.last_step = -1

    generate_warmup_scheduler1 = warmup.UntunedLinearWarmup(generate_optimizer1)
    generate_warmup_scheduler1.last_step = -1

    merge_warmup_scheduler1 = warmup.UntunedLinearWarmup(merge_optimizer1)
    merge_warmup_scheduler1.last_step = -1

elif config.warm_up_stratege == "LinearWarmup":
    encoder_warmup_scheduler =  warmup.LinearWarmup(encoder_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(encoder_optimizer)#warmup.UntunedLinearWarmup(encoder_optimizer)
    encoder_warmup_scheduler.last_step = -1 # initialize the step counter

    predict_warmup_scheduler = warmup.LinearWarmup(predict_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(predict_optimizer)
    predict_warmup_scheduler.last_step = -1

    generate_warmup_scheduler = warmup.LinearWarmup(generate_optimizer, warmup_period=config.warmup_period)#warmup.RAdamWarmup(generate_optimizer)
    generate_warmup_scheduler.last_step = -1

    merge_warmup_scheduler = warmup.LinearWarmup(merge_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(merge_optimizer)
    merge_warmup_scheduler.last_step = -1


fold=0
start_epoch=1
last_acc=0.0



# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

    encoder_1.cuda()
    predict_1.cuda()
    generate_1.cuda()
    merge_1.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])


best_acc=0

last_best_acc=0
for epoch in range(start_epoch, n_epochs):
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)

    all_len   = len(input_lengths)
    range_len = range(all_len)
             
    kl_loss_total_1 = 0
    loss_total_no_prue = 0

    kl_loss_total_2 = 0
    loss_total_prue = 0

    vae_kl1_total_1 = 0
    vae_kl1_total_2 = 0

    start = time.time()
    for idx in tqdm(range_len):#range_len:
        encoder_scheduler.step(epoch-1)
        predict_scheduler.step(epoch-1)
        generate_scheduler.step(epoch-1)
        merge_scheduler.step(epoch-1)

        encoder_warmup_scheduler.dampen()
        predict_warmup_scheduler.dampen()
        generate_warmup_scheduler.dampen()
        merge_warmup_scheduler.dampen()


        encoder_scheduler1.step(epoch-1)
        predict_scheduler1.step(epoch-1)
        generate_scheduler1.step(epoch-1)
        merge_scheduler1.step(epoch-1)

        encoder_warmup_scheduler1.dampen()
        predict_warmup_scheduler1.dampen()
        generate_warmup_scheduler1.dampen()
        merge_warmup_scheduler1.dampen()

        loss_no_prue, kl_loss1, loss_prue, kl_loss2, vae_kl1, vae_kl2 = train_tree_SWS_divide(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], 
                generate_num_ids, 
                encoder, predict, generate, merge,
                encoder_1, predict_1, generate_1, merge_1,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
                encoder_optimizer1, predict_optimizer1, generate_optimizer1, merge_optimizer1,
                output_lang, num_pos_batches[idx], is_train = True
                )
        
        loss_total_prue += loss_prue
        kl_loss_total_1 += kl_loss1

        loss_total_no_prue += loss_no_prue
        kl_loss_total_2 += kl_loss2

        vae_kl1_total_1 += vae_kl1
        vae_kl1_total_2 += vae_kl2

    # encoder_scheduler.step()
    # predict_scheduler.step()
    # generate_scheduler.step()
    # merge_scheduler.step()

    L = len(input_lengths)    
    print("loss_1:{} contra_loss_1:{} loss_2:{} contra_loss_2:{} vae_kl1_total_1:{}  vae_kl1_total_2:{} loss type:{} pool_name:{}".format(loss_total_prue / L, kl_loss_total_1 / L, loss_total_no_prue / L, kl_loss_total_2 / L, vae_kl1_total_1 / L, vae_kl1_total_2 / L, config.RDloss, config.pool_name))
    
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if (epoch-1) % test_interval == 0 or (epoch-1) > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        
        #global_unstructured_flag(parameters_to_prune, config.is_prune2test)

        for test_batch in tqdm(test_pairs):
            
            # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5]) 
            
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                    merge, output_lang, test_batch[5], beam_size=beam_size)
                                    
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1

        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        

        value_ac1 = 0
        equation_ac1 = 0
        eval_total1 = 0
        start = time.time()
    

        for test_batch in tqdm(test_pairs):
            
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder_1, predict_1, generate_1,
                                    merge_1, output_lang, test_batch[5], beam_size=beam_size)
                                    
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac1 += 1
            if equ_ac:
                equation_ac1 += 1
            eval_total1 += 1

        print(equation_ac1, value_ac1, eval_total1)
        print("test_answer_acc", float(equation_ac1) / eval_total1, float(value_ac1) / eval_total1)
        print("testing time", time_since(time.time() - start))



        print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{} loss_no_mask:{} warm_up_strategy:{} model_name:{} batch_size:{} USE_APE_word:{} quantity_num_ape:{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size, config.is_loss_no_mask, config.warm_up_stratege, config.MODEL_NAME, batch_size, config.USE_APE_word, config.quantity_num_ape))
        print("------------------------------------------------------")

        curr_acc=round(float(value_ac)/eval_total,4)
        curr_acc1=round(float(value_ac1)/eval_total1,4)
        
        curr_best = 0


        if curr_acc >= curr_acc1:
           curr_best = curr_acc
        else:
           curr_best = curr_acc1

        if curr_best>best_acc :
            last_acc = best_acc
            best_acc = curr_best
            
            torch.save(encoder.state_dict(), "models/es_t6/encoder")
            torch.save(predict.state_dict(), "models/es_t6/predict")
            torch.save(generate.state_dict(), "models/es_t6/generate")
            torch.save(merge.state_dict(), "models/es_t6/merge")

            torch.save(encoder.state_dict(), "models/es_t6/encoder_1")
            torch.save(predict.state_dict(), "models/es_t6/predict_1")
            torch.save(generate.state_dict(), "models/es_t6/generate_1")
            torch.save(merge.state_dict(), "models/es_t6/merge_1")

        else:
            print("break early stoping=================================")
            break

        # if epoch == n_epochs - 1:
        #     best_acc_fold.append((equation_ac, value_ac, eval_total))


value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()


encoder.load_state_dict(torch.load("models/es_t6/encoder"))
predict.load_state_dict(torch.load("models/es_t6/predict"))
generate.load_state_dict(torch.load("models/es_t6/generate"))
merge.load_state_dict(torch.load("models/es_t6/merge"))


for test_batch in valid_pairs:
    # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5]) 
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")



value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()


encoder_1.load_state_dict(torch.load("models/es_t6/encoder_1"))
predict_1.load_state_dict(torch.load("models/es_t6/predict_1"))
generate_1.load_state_dict(torch.load("models/es_t6/generate_1"))
merge_1.load_state_dict(torch.load("models/es_t6/merge_1"))

# global_unstructured_flag(parameters_to_prune, config.is_prune2test)#False)

for test_batch in valid_pairs:
    # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5]) 
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder_1, predict_1, generate_1,
                                     merge_1, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")
