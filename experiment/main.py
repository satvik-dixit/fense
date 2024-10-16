import json
import numpy as np
import pandas as pd
import json 
import torch
from tqdm import tqdm
import sys
sys.path.append('../caption-evaluation-tools')
from sentence_transformers import SentenceTransformer, CrossEncoder
# from bert_score import BERTScorer
# from bleurt import score as bleurt_score
from eval_metrics import evaluate_metrics_from_lists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataloader import get_former, get_latter
from msclap import CLAP

model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device='cuda:0')
model_sb.eval()

clap_model = CLAP(version='2023', use_cuda=True)  # Assuming you want to use GPU

def cosine_similarity(input, target):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()

def get_text_score(all_preds_text, all_refs_text, method='sentence-bert', average=True, audio_files=None, dataset_name='audiocaps'):
    N = len(all_preds_text)
    K = len(all_refs_text[0])
    all_preds_text = np.array(all_preds_text, dtype=str)
    all_refs_text = np.array(all_refs_text, dtype=str)

    print('all_preds_text shape:', len(all_preds_text))
    print('all_refs_text shape:', len(all_refs_text))
    print('all_refs_text shape:', len(all_refs_text[0]))

    if method=='sentence-bert' or method=='ms-CLAP':
        score = torch.zeros((N, K))
    else:
        score = torch.zeros((N, 1))

    # For Sentence-BERT
    if method == 'sentence-bert':
        preds_sb = torch.Tensor(model_sb.encode(all_preds_text))
        refs_sb = torch.Tensor(np.array([model_sb.encode(x) for x in all_refs_text]))
        for i in range(K):
            score[:,i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_sb, refs_sb[:,i])])
    
   # For CLAP
    elif method == 'ms-CLAP':
        # preds_clap = clap_model.get_text_embeddings(all_preds_text.tolist()).to('cuda')
        # print('shape 1:', preds_clap.shape)
        preds_clap = torch.stack([clap_model.get_text_embeddings([pred]).to('cuda') for pred in all_preds_text], dim=0).squeeze()
        print('preds_clap shape:', preds_clap.shape)
        refs_clap = torch.stack([clap_model.get_text_embeddings(refs).to('cuda') for refs in all_refs_text], dim=0)
        print('refs_clap shape:', refs_clap.shape)
        for i in range(K):
            score[:, i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_clap, refs_clap[:, i])])
        print('score:', score)

    # For Audio CLAP
    elif method == 'ms_clap_audio_caption':
        if audio_files is None:
            raise ValueError("Audio files must be provided for ms_clap_audio_caption.")
        # preds_clap = clap_model.get_text_embeddings(all_preds_text.tolist()).to('cuda')
        preds_clap = torch.stack([clap_model.get_text_embeddings([pred]).to('cuda') for pred in all_preds_text], dim=0).squeeze()
        print('preds_clap shape:', preds_clap.shape)
        if dataset_name=='clotho':
            audio_files = [f'/content/fense/clotho_eval_audio/{audio}' for audio in audio_files]
        elif dataset_name=='audiocaps':
            audio_files = [f'/content/fense/audiocaps_caption_eval/{audio}.wav' for audio in audio_files]
        print('audio_files:', audio_files)
        # audio_embs = [clap_model.get_audio_embeddings(audio_files)]
        audio_embs = torch.stack([clap_model.get_audio_embeddings([audio_file]).to('cuda') for audio_file in audio_files], dim=0)
        # audio_embs = audio_embs.squeeze()
        print('audio_embs shape:', audio_embs.shape)
        for i in range(1):
            score[:, i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_clap, audio_embs[:, i])])
        print('score:', score)
        print('score shape:', score.shape)


    # Calculate average or max score
    score = score.mean(dim=1) if average else score.max(dim=1)[0]
    print('avg_score:', score)
    print('avg_score shape:', score.shape)

    return score

def get_accuracy(machine_score, human_score, threshold=0):
    cnt = 0
    # threshold = 0.001*np.average([abs(t) for t in machine_score])
    # threshold = 1e-6
    N = np.sum([x!=0 for x in human_score]) if threshold==0 else len(human_score)
    for i, (ms, hs) in enumerate(zip(machine_score, human_score)):
        if ms*hs > 0 or abs(ms-hs) < threshold:
            cnt += 1
    return cnt / N

import csv

# def print_accuracy(machine_score, human_score, csv_filename="accuracy_results.csv"):
#     results = []
#     data_to_save = []  # Prepare a list for data to save in CSV

#     for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
#         if facet != 'MM':
#             sub_score = machine_score[i*250:(i+1)*250]
#             sub_truth = human_score[i*250:(i+1)*250]
#         else:
#             sub_score = machine_score[i*250:]
#             sub_truth = human_score[i*250:]

#         acc = get_accuracy(sub_score, sub_truth)
#         results.append(round(acc * 100, 1))
        
#         # Print the accuracy for each facet
#         print(facet, "%.1f" % (acc * 100))

#         # Store the data for each facet in the list for CSV
#         for score, truth in zip(sub_score, sub_truth):
#             data_to_save.append([facet, score, truth])

#     # Calculate total accuracy and print it
#     acc = get_accuracy(machine_score, human_score)
#     results.append(round(acc * 100, 1))
#     print("total acc: %.1f" % (acc * 100))

#     # Save the data to CSV file
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['facet', 'subscore', 'subtruth'])  # CSV header
#         writer.writerows(data_to_save)  # Write all data rows

#     return results

def print_accuracy(machine_score, human_score):
    results = []
    for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
        if facet != 'MM':
            sub_score = machine_score[i*250:(i+1)*250]
            sub_truth = human_score[i*250:(i+1)*250]
        else:
            sub_score = machine_score[i*250:]
            sub_truth = human_score[i*250:]
        acc = get_accuracy(sub_score, sub_truth)
        results.append(round(acc*100, 1))
        print(facet,  "%.1f" % (acc*100))
    acc = get_accuracy(machine_score, human_score)
    results.append(round(acc*100, 1))
    print("total acc: %.1f" % (acc*100))
    return results


if __name__ == '__main__':
    for dataset in ['audiocaps', 'clotho']:
        score, score0, score1 = {}, {}, {}
        mm_score, mm_score0, mm_score1 = {}, {}, {}

        hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth, hh_audio_files = get_former(dataset)
        print('hh_audio_files', hh_audio_files)
        mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth, mm_audio_files = get_latter(dataset)
        print('mm_audio_files', mm_audio_files)

        # Iterate through both embedding methods: Sentence-BERT and CLAP and CLAP_audio_caption
        for metric in ['ms-CLAP', 'sentence-bert', 'ms_clap_audio_caption']:
            score0[metric] = get_text_score(hh_preds_text0, hh_refs_text0, metric, audio_files=hh_audio_files, dataset_name=dataset)
            score1[metric] = get_text_score(hh_preds_text1, hh_refs_text1, metric, audio_files=hh_audio_files, dataset_name=dataset)

            mm_score0[metric] = get_text_score(mm_preds_text0, mm_refs_text, metric, audio_files=mm_audio_files, dataset_name=dataset)
            mm_score1[metric] = get_text_score(mm_preds_text1, mm_refs_text, metric, audio_files=mm_audio_files, dataset_name=dataset)

        total_score0, total_score1, total_score = {}, {}, {}
        for metric in score0:
            total_score0[metric] = torch.cat([score0[metric], mm_score0[metric]])
            total_score1[metric] = torch.cat([score1[metric], mm_score1[metric]])
            total_score[metric] = total_score0[metric] - total_score1[metric]
        total_human_truth = hh_human_truth + mm_human_truth

        metrics0, per_file_metrics0 = evaluate_metrics_from_lists(hh_preds_text0, hh_refs_text0)
        metrics1, per_file_metrics1 = evaluate_metrics_from_lists(hh_preds_text1, hh_refs_text1)

        # expand the references (choose 4 from 5)
        mm_preds_text0_exp = [x for x in mm_preds_text0 for i in range(5)]
        mm_preds_text1_exp = [x for x in mm_preds_text1 for i in range(5)]
        mm_refs_text_exp = []
        for refs in mm_refs_text:
            for i in range(5):
                mm_refs_text_exp.append([v for k,v in enumerate(refs) if k%5!=i])

        mm_metrics0, mm_per_file_metrics0 = evaluate_metrics_from_lists(mm_preds_text0_exp, mm_refs_text_exp)
        mm_metrics1, mm_per_file_metrics1 = evaluate_metrics_from_lists(mm_preds_text1_exp, mm_refs_text_exp)

        def get_score_list(per_file_metric, metric):
            if metric == 'SPICE':
                return [v[metric]['All']['f'] for k,v in per_file_metric.items()]
            else:
                return [v[metric] for k,v in per_file_metric.items()]

        def shrink(arr, repeat=5):
            return np.array(arr).reshape(-1, repeat).mean(axis=1).tolist()

        baseline_list = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr', 'SPICE', 'SPIDEr']
        for metric in baseline_list:
            total_score0[metric] = torch.Tensor(get_score_list(per_file_metrics0, metric) + shrink(get_score_list(mm_per_file_metrics0, metric)))
            total_score1[metric] = torch.Tensor(get_score_list(per_file_metrics1, metric) + shrink(get_score_list(mm_per_file_metrics1, metric)))
            total_score[metric] = total_score0[metric] - total_score1[metric]

        results = []
        for metric in total_score:
            print(metric)
            tmp = print_accuracy(total_score[metric], total_human_truth)
            results.append(tmp)

        df = pd.DataFrame(results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        df.index = [x for x in total_score]
        df.to_csv('results_{}.csv'.format(dataset))

        ##############################################################################
        # Error detector applied
        ##############################################################################

        # load pre-computed ndarray 
        probs0 = np.load('../bert_for_fluency/cache/probs0_alltrain_{}.npy'.format(dataset))
        probs1 = np.load('../bert_for_fluency/cache/probs1_alltrain_{}.npy'.format(dataset))

        coef = 0.9
        thresholds = np.arange(0.0, 1.05, 0.05)
        results_df = pd.DataFrame(columns=['Method', 'Threshold', 'HC', 'HI', 'HM', 'MM', 'Total'])

        for method in total_score:
            for thres in thresholds:
                score_penalty = [s1-s1*coef*(p1>thres)-(s2-s2*coef*(p2>thres)) for s1,s2,p1,p2 in zip(total_score0[method],total_score1[method],probs0[:,-1],probs1[:,-1])]
                
                print(f"Method: {method}, Threshold: {thres:.2f}")
                tmp = print_accuracy(score_penalty, total_human_truth)
                
                results_df = results_df.append({
                    'Method': method,
                    'Threshold': thres,
                    'HC': tmp[0],
                    'HI': tmp[1],
                    'HM': tmp[2],
                    'MM': tmp[3],
                    'Total': tmp[4]
                }, ignore_index=True)

        results_df.to_csv(f'fluency_varied_thresholds_{dataset}.csv', index=False)


        # # load pre-computed ndarray 
        # probs0 = np.load('../bert_for_fluency/cache/probs0_alltrain_{}.npy'.format(dataset))
        # probs1 = np.load('../bert_for_fluency/cache/probs1_alltrain_{}.npy'.format(dataset))

        # score_penalty = {}
        # thres = 0.9
        # coef = 0.9

        # for method in total_score:
        #     score_penalty[method] = [s1-s1*coef*(p1>thres)-(s2-s2*coef*(p2>thres)) for s1,s2,p1,p2 in zip(total_score0[method],total_score1[method],probs0[:,-1],probs1[:,-1])]

        # results = []
        # for method in score_penalty:
        #     print(method)
        #     tmp = print_accuracy(score_penalty[method], total_human_truth)
        #     results.append(tmp)

        # df = pd.DataFrame(results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        # df.index = [x for x in total_score]
        # df.to_csv('fluency_{}.csv'.format(dataset))
