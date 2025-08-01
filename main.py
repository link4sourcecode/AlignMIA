import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from rouge import Rouge
import torch.nn.functional as F
from data_process import extract_nonmemdata, eval_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='checkpoints/LLaVA-7B-v1.5', help='Path to the model checkpoint')
parser.add_argument('--patch_size', type=int, default=14, help='Patch size for masking')
parser.add_argument('--random_times', type=int, default=10, help='Number of random sampling times')
parser.add_argument('--rate_list', type=float, nargs='+', default=[0.0, 0.25, 0.50, 0.75], help='List of masking rates')
parser.add_argument('--save_dir', type=str, default='./model_outputs', help='Prefix for saving output files')
parser.add_argument('--gen_len', type=int, default=50, help='Length of generated text')
parser.add_argument('--task_type', type=str, default='short', help='Task type: short or long')
parser.add_argument('--train_num', type=int, default=1000, help='Number of training samples')
parser.add_argument('--train_seed', type=int, default=42, help='Seed for training data shuffling')
parser.add_argument('--val_nonmemnum', type=int, default=500, help='Number of validation samples for nonmember data')
parser.add_argument('--val_ratio', type=float, default=1.0, help='Ratio of val-member data to val-nonmember data')
parser.add_argument('--val_seed', type=int, default=42, help='Seed for validation data shuffling')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--local_seed', type=int, default=42, help='Local random seed for sampling')
parser.add_argument('--image_size', type=int, default=None, help='Size of the input images (e.g.336x336). Determines the total number of patches that can be masked.')
args = parser.parse_args()

if "llava" in args.model_path.lower():
    from llava.model_utils import *
elif "mgm" in args.model_path.lower():
    from mgm.model_utils import *
elif "minigpt4" in args.model_path.lower():
    from minigpt4.model_utils import *
else:
    raise ValueError("Unsupported model type. Please specify a valid model path.")

def randomrate_sample(local_random, data, rate):
    sample_size = int(len(data) * rate)
    local_random.shuffle(data)
    return data, data[:sample_size]
def maskimg_outputtxt(model, tokenizer, image_processor, alldata, random_times, rate, device, local_random, init_imgnum=576, mode='prenonmem', gen_len=50, patch_size=14, save_dir='./model_outputs'):
    init_imgidx = list(range(init_imgnum))
    for idx, item in enumerate(tqdm(alldata, desc="Processing")):
        data_maskimgtoken_txt = []
        if rate == 0:
            random_times = 1
        for i in range(random_times):
            init_imgidx, padimglist = randomrate_sample(local_random, init_imgidx, rate)
            masked_image = process_images([item['image_path']], image_processor, model.config, padimglist, patch_size)[0]
            generated_text = generate_output(model, tokenizer, image_processor, masked_image, item['question'], gen_len=gen_len)
            data_maskimgtoken_txt.append(generated_text)
        if os.path.exists(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputtxt.pt'):
            alldata_maskimgtoken_txt = torch.load(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputtxt.pt')
        else:
            os.makedirs(save_dir, exist_ok=True)
            alldata_maskimgtoken_txt = []
        alldata_maskimgtoken_txt.append(data_maskimgtoken_txt)
        torch.save(alldata_maskimgtoken_txt, save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputtxt.pt')
        del alldata_maskimgtoken_txt
def maskimg_gtloss_entropy(model, tokenizer, image_processor, alldata, random_times, rate, device, local_random, init_imgnum=576, mode='prenonmem', patch_size=14, save_dir='./model_outputs'):
    init_imgidx = list(range(init_imgnum))
    for idx, item in enumerate(tqdm(alldata, desc="Processing")):
        data_maskimgtoken_loss = []
        data_maskimgtoken_entro = []
        if rate == 0:
            random_times = 1
        for i in range(random_times):
            init_imgidx, padimglist = randomrate_sample(local_random, init_imgidx, rate)
            masked_image = process_images([item['image_path']], image_processor, model.config, padimglist, patch_size)[0]
            loss, logits, answer_len = compute_loss(model, tokenizer, image_processor, masked_image, [item['answer']], device, item['question'])
            all_prob = F.softmax(logits[0,-answer_len:,:].float(), dim=-1)
            entro = -torch.sum(all_prob * torch.log(all_prob + 1e-9), dim=-1)
            entro = torch.mean(entro)
            data_maskimgtoken_loss.append(loss.item())
            data_maskimgtoken_entro.append(entro.item())
            del loss, logits, answer_len, all_prob
            torch.cuda.empty_cache()

        if os.path.exists(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputlosses.pt'):
            alldata_maskimgtoken_loss = torch.load(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputlosses.pt')
        else:
            os.makedirs(save_dir, exist_ok=True)
            alldata_maskimgtoken_loss = []
        if os.path.exists(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputentropy.pt'):
            alldata_maskimgtoken_entro = torch.load(save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputentropy.pt')
        else:
            os.makedirs(save_dir, exist_ok=True)
            alldata_maskimgtoken_entro = []
        alldata_maskimgtoken_entro.append(data_maskimgtoken_entro)
        alldata_maskimgtoken_loss.append(data_maskimgtoken_loss)
        torch.save(alldata_maskimgtoken_entro, save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputentropy.pt')
        torch.save(alldata_maskimgtoken_loss, save_dir + f'/{mode}_maskinimgtoken{int(rate * 10):02d}_outputlosses.pt')
        del alldata_maskimgtoken_loss, alldata_maskimgtoken_entro
def list_check(alist):
    for i in range(len(alist)):
        for j in range(len(alist[i])):
            if type(alist[i][j]) is list:
                alist[i][j] = alist[i][j][0]
    return alist
def get_fea(mode, fea, model_name, rate_list=[0.0, 0.25, 0.5, 0.75], suffix='output'):
    nonmem_fea_dict = {}
    for rate in rate_list:
        rate_key = f"{int(rate * 10):02d}"
        file_path = args.save_dir + f'/{mode}_maskinimgtoken{rate_key}_{suffix}{fea}.pt'
        nonmem_fea_dict[rate] = torch.load(file_path)

    # sometimes minigpt4 will generate lists, we need to flatten it
    if 'minigpt4' in model_name.lower():
        for rate in nonmem_fea_dict:
            nonmem_fea_dict[rate] = list_check(nonmem_fea_dict[rate])
    
    for rate in nonmem_fea_dict:
        nonmem_fea_dict[rate] = np.array(nonmem_fea_dict[rate])
    if 0.0 in rate:
        nonmem_fea_dict[0.0] = nonmem_fea_dict[0.0].reshape(nonmem_fea_dict[0.0].shape[0], -1)
        if nonmem_fea_dict[0.0].shape[-1] == 1:
            nonmem_fea_dict[0.0] = np.tile(nonmem_fea_dict[0.0], (1, nonmem_fea_dict[rate_list[1]].shape[-1]))

    nonmem_fea_list = []
    for rate in rate_list:
        if rate in nonmem_fea_dict:
            nonmem_fea_list.append(nonmem_fea_dict[rate])
        else:
            print(f"Warning: Rate {rate} not found in loaded data")
    if nonmem_fea_list:
        nonmem_fea = np.stack(nonmem_fea_list, axis=1)
        return nonmem_fea
    else:
        raise ValueError("No valid feature data found")
def txtscore_compute(references, candidate):
    rouge1 = []
    rouge2 = []
    rougel = []
    bleu_score = []
    cand = ' '.join(candidate)
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    for reference in references:
        ref = ' '.join(reference)
        if reference[0] == '':
            rouge1.append(0)
            rouge2.append(0)
            rougel.append(0)
            bleu_score.append(0)
            continue
        try:
            rouge_score = rouge.get_scores(hyps=cand, refs=ref)
            rouge1.append(rouge_score[0]["rouge-1"]['f'])
            rouge2.append(rouge_score[0]["rouge-2"]['f'])
            rougel.append(rouge_score[0]["rouge-l"]['f'])
            bleu_score.append(sentence_bleu([ref], cand, smoothing_function=smooth))
        except ValueError:
            return [rouge1, rouge2, rougel, bleu_score]
    return [rouge1, rouge2, rougel, bleu_score]
def get_txt_score(evaldata_txt, evaldata):
    scores = []
    for idx in range(len(evaldata)):
        references = [[sentence] for sentence in evaldata_txt[idx]]
        scores.append(txtscore_compute(references, [evaldata[idx]]))
    scores = np.array(scores)
    return scores
def get_score(gt, mode=('prenonmem_14', 'eval_14'), fea=('losses', 'entropy', 'txt'), model_name='llava', rate_list=[0.0, 0.25, 0.5, 0.75], suffix='output'):
    prenonmem_fea = {}
    eval_fea = {}
    all_scores = {}
    for fea_ in fea:
        prenonmem_fea[fea_] = get_fea(mode[0], fea_, model_name, rate_list, suffix)
        eval_fea[fea_] = get_fea(mode[1], fea_, model_name, rate_list, suffix)
    gt = np.array(gt)
    if 'txt' in fea:
        prenonmem_mem_nonmem_txt = np.concatenate([prenonmem_fea['txt'], eval_fea['txt']], axis=0).reshape(len(gt), -1)
        all_scores['txt'] = get_txt_score(prenonmem_mem_nonmem_txt, gt)
        all_scores['txt'] = all_scores['txt'].transpose(0, 2, 1).\
            reshape(len(gt), prenonmem_fea['txt'].shape[1], prenonmem_fea['txt'].shape[2], 4)
    for fea_ in fea:
        if fea_ == 'txt':
            continue
        all_scores[fea_] = np.concatenate([prenonmem_fea[fea_], eval_fea[fea_]], axis=0)
        all_scores[fea_] = np.expand_dims(all_scores[fea_], axis=-1)

    prenonmem_mem_nonmem_score = []
    for fea_ in fea:
        prenonmem_mem_nonmem_score.append(all_scores[fea_])
    prenonmem_mem_nonmem_score = np.concatenate(prenonmem_mem_nonmem_score, axis=-1)
    return prenonmem_mem_nonmem_score

if __name__ == '__main__':
    model, tokenizer, image_processor, context_len = load_targetmodel(args.model_path)
    nonmem_data, residue_data = extract_nonmemdata(args.task_type, args.train_seed, args.train_num)
    eval_memdataset, eval_nonmemdataset= eval_dataset(args.task_type, args.train_num, args.train_seed, args.val_nonmemnum, 
                                                                args.val_ratio, args.val_seed)
    print("load model and data done!")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_random = random.Random()
    local_random.seed(args.local_seed)
    eval_data = eval_memdataset + eval_nonmemdataset
    if args.image_size is None:
        if "llava" in args.model_path.lower():
            args.image_size = image_processor.crop_size['height'] * image_processor.crop_size['width']
        elif "mgm" in args.model_path.lower():
            args.image_size = model.config.image_size_aux ** 2
        else:
            args.image_size = 224**2
    patch_num = args.image_size // (args.patch_size * args.patch_size)

    # mask images and generate model outputs
    for rate in args.rate_list:
        print(f"---------rate: {rate}-------------")
        maskimg_gtloss_entropy(model, tokenizer, image_processor, nonmem_data, args.random_times, rate, device, local_random, patch_num, f'prenonmem_{args.patch_size}', args.patch_size, args.save_dir)
        maskimg_gtloss_entropy(model, tokenizer, image_processor, eval_data, args.random_times, rate, device, local_random, patch_num, f'eval_{args.patch_size}', args.patch_size, args.save_dir)
        maskimg_outputtxt(model, tokenizer, image_processor, nonmem_data, args.random_times, rate, device, local_random, patch_num, f'prenonmem_{args.patch_size}', args.gen_len, args.patch_size, args.save_dir)
        maskimg_outputtxt(model, tokenizer, image_processor, eval_data, args.random_times, rate, device, local_random, patch_num, f'eval_{args.patch_size}', args.gen_len, args.patch_size, args.save_dir)

    # calculate mia score
    nonmem_gt = [item['answer'] for item in nonmem_data]
    eval_gt = [item['answer'] for item in eval_data]
    all_gt = nonmem_gt + eval_gt
    prenonmem_mem_nonmem_score = get_score(all_gt, mode=(f'prenonmem_{args.patch_size}', f'eval_{args.patch_size}'), model_name=os.path.basename(args.model_path), rate_list=args.rate_list, suffix='output')
    np.save(os.path.join(args.save_dir, f'mia_score.npy'), prenonmem_mem_nonmem_score)
