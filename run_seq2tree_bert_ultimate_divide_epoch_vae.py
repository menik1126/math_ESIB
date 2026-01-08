# coding: utf-8
from src.train_and_evaluate_divide_vae import *
from src.models_vae_divide import *
import time
import torch.optim
import argparse

from src.expressions_transfer import *
from tqdm import tqdm

from src import config
import torch.nn.utils.prune as prune
import pytorch_warmup as warmup


def parse_args():
    parser = argparse.ArgumentParser(description='Math23K Full Model (VAE + Dual Network)')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden layer dimension')
    parser.add_argument('--embedding_size', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--n_epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--model_name', type=str, default='roberta', help='Pretrained model name')
    parser.add_argument('--use_ape', type=str, default='false', help='Use APE dataset (true/false)')
    parser.add_argument('--warmup_period', type=int, default=3000, help='Warmup steps')
    parser.add_argument('--contra_weight', type=float, default=0.005, help='Contrastive loss weight')
    parser.add_argument('--test_interval', type=int, default=5, help='Test interval (epochs)')
    return parser.parse_args()


def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


    


def get_train_test_fold(ori_path, prefix, data, pairs):

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
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold



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
for d in range(config.quantity_num):
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

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)



temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(config.ori_path, config.prefix, data, pairs)
best_acc_fold = []


pairs_trained = train_fold
pairs_tested = test_fold
pairs_validated = valid_fold
    

input_lang, output_lang, train_pairs, (valid_pairs, test_pairs) = prepare_data(tokenizer, pairs_trained, [pairs_validated, pairs_tested], 5, generate_nums,
                                                            copy_nums, tree=True)

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

# the embedding layer is  only for generated number embeddings, operators, and paddings
encoder_optimizer1 = torch.optim.AdamW(encoder_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
predict_optimizer1 = torch.optim.AdamW(predict_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
generate_optimizer1 = torch.optim.AdamW(generate_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
merge_optimizer1 = torch.optim.AdamW(merge_1.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)




encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
predict_scheduler = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
generate_scheduler = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
merge_scheduler = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer, milestones=[config.n_epochs//3], gamma=0.1)


encoder_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer1, milestones=[config.n_epochs//3], gamma=0.1)
predict_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer1, milestones=[config.n_epochs//3], gamma=0.1)
generate_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer1, milestones=[config.n_epochs//3], gamma=0.1)
merge_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer1, milestones=[config.n_epochs//3], gamma=0.1)

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
alternate_flag = True
for epoch in range(1, n_epochs+1):
    
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()

    all_len   = len(input_lengths)
    range_len = range(all_len)
             
    kl_loss_total_1 = 0
    loss_total_no_prue = 0

    kl_loss_total_2 = 0
    loss_total_prue = 0

    vae_loss_t1 = 0
    vae_loss_t2 = 0

    alternate_flag = bool(1 -alternate_flag)
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


        loss_no_prue, kl_loss1, loss_prue, kl_loss2, vae_loss1, vae_loss2 = train_tree_SWS_divide(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], 
                generate_num_ids, 
                encoder, predict, generate, merge,
                encoder_1, predict_1, generate_1, merge_1,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
                encoder_optimizer1, predict_optimizer1, generate_optimizer1, merge_optimizer1,
                output_lang, num_pos_batches[idx], is_train = True,
                alternate_flag = alternate_flag)


        
        
        
        loss_total_prue += loss_prue
        kl_loss_total_1 += kl_loss1

        loss_total_no_prue += loss_no_prue
        kl_loss_total_2 += kl_loss2

        vae_loss_t1+=vae_loss1
        vae_loss_t2+=vae_loss2


    L = len(input_lengths)
    print("loss_1:{} contra_loss_1:{} loss_2:{} contra_loss_2:{} vae_1:{} vae_2:{} loss type:{}".format(loss_total_prue / L, kl_loss_total_1 / L, loss_total_no_prue / L, kl_loss_total_2 / L, vae_loss_t1/L, vae_loss_t2/L,config.RDloss))
    

    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if (epoch-1) % test_interval == 0 or (epoch-1) > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        
        for test_batch in tqdm(valid_pairs):
            
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

        print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{}  warm_up_strategy:{} model_name:{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size, config.warm_up_stratege, config.MODEL_NAME))
        print("------------------------------------------------------")
        # torch.save(encoder.state_dict(), "models/encoder")
        # torch.save(predict.state_dict(), "models/predict")
        # torch.save(generate.state_dict(), "models/generate")
        # torch.save(merge.state_dict(), "models/merge")
        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))

value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()



for test_batch in test_pairs:
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc0", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")


value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()


for test_batch in test_pairs:
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder_1, predict_1, generate_1,
                                     merge_1, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc1", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")
