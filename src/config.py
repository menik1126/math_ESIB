
Max_Question_len=120
Max_Expression_len=50

forget_rate = 0.2
exponent = 1
num_gradual = 10


weight_decay = 1e-5
beam_size = 5
n_layers = 2
batch_size = 16
# MODEL_NAME='distilbert-base-multilingual-cased'
MODEL_NAME="roberta"#"xlnet"#'bert-base-chinese'#"roberta"#"xml-roberta" 'xml-roberta-base'#"roberta"
warm_up_stratege = "original"#"UntunedExponentialWarmup"#'RAdamWarmup'#"original"#"ExponentialWarmup"#"RAdamWarmup"#'LinearWarmup'
pool_name = "cls_pool"#"cls_pool" "mean_pool" "conv_pool"
data_file = "data/Math_23K.json"
warmup_period = 3000


is_train_kl = True
embedding_size = 1024#1024#512#1024
hidden_size = 1024#1024#1024

is_RDrop = True
learning_rate = 5e-5#5e-5
is_prune2test = False
quantity_num= 15 #15
quantity_num_ape = 28#22
prunePercent = 0.2 #0.2
contra_weight = 0.005#0.005
CEB_weight = 0.005

is_CEB_detached = False
is_v = False
is_Cross = False

is_vae = True

latent_dim = 50#50
num_filters=256
filter_size=(2,3,4)

is_prune = True
is_graph = False
is_loss_no_mask  = False
is_mid_loss = False

USE_APE = False#True

#APE  word  彩电 是 两种 电视机
#APE  char 彩 电 是 两 种 电 视 机
USE_APE_word=False
USE_APE_char=True

RDloss = 'kl_loss'#'cosine_loss'#'wasserstein_loss'#'nt_xent_loss'#'kl_loss'

dropout = 0.5
n_epochs = 80
is_mask = True

is_exposure_bias = False
is_merge = False


is_em_dropout = False





test_interval = 5

ori_path = './data/'
prefix = '23k_processed.json'

# TEACHER_FORCE_RATE = 0.5
# MODE = ['Linear']
# MODE = ['Exponential', 0.999]
MODE = ['Inverse_Sigmoid', 15000]
# 这个噪音是啥
greed_gumbel_noise = 0.5
epsilon = 1e-20

# Early stop
patience_epoch = 3

# pruning
pruning_method = 'topK'
mask_init  = 'constant'
mask_scale = 0.
temperature = -0.2



