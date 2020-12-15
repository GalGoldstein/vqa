import json
import pickle

with open('data/v2_mscoco_val2014_annotations.json') as f:
    val_annot = json.load(f)

with open('data/v2_OpenEnded_mscoco_val2014_questions.json') as f:
    val_quest = json.load(f)

val_target = pickle.load(open("data/cache/val_target.pkl", "rb"))
ans2label = pickle.load(open("data/cache/trainval_ans2label.pkl", "rb"))
label2ans = pickle.load(open("data/cache/trainval_label2ans.pkl", "rb"))
