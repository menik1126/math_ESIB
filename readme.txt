Check the accuracy of CN and SN on APE210K:
1.set the USE_APE = True in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run  run_seq2tree_APE_early_SP_VAE.py


Check the accuracy of CN and SN on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_divide_epoch_vae.py

Check the accuracy of CN w/o MT on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_comp_vae.py

Check the accuracy of CN w/o VIB on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_divide_epoch.py

Check the accuracy of CN w/o MT+VIB on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_comp.py

Check the accuracy of CN(bert) w/o MT+VIB on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='bert-base-chinese' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_comp.py

Check the accuracy of CN(bert) w/o MT+VIB on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='bert-base-chinese' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_comp_vae.py

Check the accuracy of CN with V_sdl on Math23k:
1.set the USE_APE = False in math_ESIB/src/config.py
2.set the MODEL_NAME='roberta' in math_ESIB/src/config.py
3.please run run_seq2tree_bert_ultimate_divide_dice.py






