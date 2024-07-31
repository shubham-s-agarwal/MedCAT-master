from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

phase_num = 0
config_dict = {"general":{"category_name":"Experiencer","cntx_left":40,"cntx_right":40,"tokenizer_name":"bert-tokenizer","category_value2id":{"Family":1,"Other":0,"Patient":2}},"model":{"input_size":768,"hidden_size":200,"model_name":"bert","nclasses":3,"dropout":0.4,"model_freeze_layers":False,"phase_number":0,"category_undersample":"Other","num_layers":3},"train":{"nepochs":40,"auto_save_model":False,"batch_size":32,"lr":5e-4,"class_weights":[0.5,0.5],"metric":{"base":"macro avg","score":"f1-score"}}}
# config_dict = {"general":{"category_name":"Status","cntx_left":20,"cntx_right":15,"tokenizer_name":"bert-tokenizer","category_value2id":{"Other":1,"Confirmed":0}},"model":{"model_name":"bert","nclasses":2,"phase_number":phase_num,"category_undersample":"Other","num_layers":3},"train":{"nepochs":40,"auto_save_model":False,"metric":{"base":"macro avg","score":"f1-score"}}}
config = ConfigMetaCAT()

print(config)
config.merge_config(config_dict)

tokenizer = TokenizerWrapperBERT.load("Status", "bert-base-uncased")

import json
json_files = []
for file in os.listdir(r'./data/'):
  if file.endswith(".json") and 'MedCAT_Export' in file:
    json_files.append("./data/"+file)

mc = MetaCAT(tokenizer=tokenizer, embeddings=None, config=config)
formatted_data = mc.train(json_files, save_dir_path="Dummy")

with open("formatted_data_exp_2.txt",'w') as f:
  f.write(str(formatted_data))
  f.close()

# print("\n\nWinner Report:",results)

# print("\n**************\nPRINTING CONFUSION MATRIX FOR TEST DATASET")
# cm = results['confusion_matrix']
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={"Confirmed": 0,"Other": 1})
# disp.plot()
# plt.show()
