import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import random
import math
from tqdm import tqdm
import shutil

def json_to_list(output_file):
    with open(output_file, 'r') as f:
        sample_data = json.load(f)
    return sample_data
def extract_nonmemdata(type, seed=42, num=2000):
    vqav2_data = json_to_list('./data/vqav2/vqav2_valsamples.json')
    vizwiz_data = json_to_list('./data/Vizwiz/vizwiz_valsamples.json')
    imagenet1k_data = json_to_list('./data/ImageNet1K/imagenet1kval_samples.json')
    q_bench_data = json_to_list('./data/Q_Bench/qbench_samples.json')
    pope_data = json_to_list('./data/POPE/pope_samples.json')
    mme_data = json_to_list('./data/MME/mme_samples.json')

    flickr8k_data = json_to_list('./data/Flickr8k/flickr8k_samples.json')
    coco2017_data = json_to_list('./data/COCO2017/cococaptions_valsamples.json')
    llava_data = json_to_list('./data/llava_bench_in_wild/llavabenchinthewild_captionsamples.json')
    llava_gpt4_data = json_to_list('./data/llava_bench_in_wild/llavabenchinthewild_gpt4samples.json')
    if type == 'short':
        split_num = math.ceil(num / 6)
        random.seed(seed)
        random.shuffle(vqav2_data)
        random.shuffle(vizwiz_data)
        random.shuffle(imagenet1k_data)
        random.shuffle(q_bench_data)
        random.shuffle(pope_data)
        random.shuffle(mme_data)
        data_list = vqav2_data[:split_num] + vizwiz_data[:split_num] + q_bench_data[:split_num] + pope_data[:split_num] + mme_data[:split_num]
        len_residue = num - len(data_list)
        data_list += imagenet1k_data[:len_residue]
        residue_data = vqav2_data[split_num:] + vizwiz_data[split_num:] + q_bench_data[split_num:] + pope_data[split_num:] + mme_data[split_num:] + imagenet1k_data[len_residue:]
    elif type == 'long':
        split_num = math.ceil(num / 4)
        random.seed(seed)
        random.shuffle(flickr8k_data)
        random.shuffle(coco2017_data)
        random.shuffle(llava_data)
        random.shuffle(llava_gpt4_data)
        data_list = llava_data[:split_num] + llava_gpt4_data[:split_num]
        len_residue_split = math.ceil((num - len(data_list)) / 2)
        data_list = data_list + coco2017_data[:len_residue_split]
        len_residue = num - len(data_list)
        data_list += flickr8k_data[:len_residue]
        residue_data = llava_data[split_num:] + llava_gpt4_data[split_num:] + coco2017_data[len_residue_split:] + flickr8k_data[len_residue:]
    return data_list, residue_data
def eval_dataset(type, train_nonmemnum=2000, train_seed=42, val_nonmemnum=1000, ratio=1.0, val_seed=42):
    residue_data = extract_nonmemdata(type, train_seed, train_nonmemnum)[1]
    random.seed(val_seed)
    random.shuffle(residue_data)
    val_nonmemdata = residue_data[:val_nonmemnum]
    if type == 'short':
        with open('./data/mem_shortdata.json') as f:
            mem_data = json.load(f)
        val_memdata = mem_data[:int(val_nonmemnum * ratio)]
    elif type == 'long':
        with open('./data/mem_longdata.json') as f:
            mem_data = json.load(f)
        val_memdata = mem_data[:int(val_nonmemnum * ratio)]
    return val_memdata, val_nonmemdata

def extract_memdata(data_path, task_type='short', data_type=None, max_num=10000, extract_num=2000, image_root='./data/images', save_dir='./data'):
    with open(data_path, 'r') as f:
        data = json.load(f)
    image_questions = []
    num = 0
    for item in tqdm(data, desc='Extracting ftdata'):
        if "image" not in item:
            continue
        image_path = item["image"]
        if data_type not in image_path and data_type is not None:
            continue
        conversations = item['conversations']
        for i in range(0, len(conversations), 2):
            question = conversations[i]['value'].replace("<image>\n", "").replace("\n<image>", "")
            answer = conversations[i + 1]['value'] if i + 1 < len(conversations) else "No answer available"
            words = answer.split() 
            if task_type == 'short':
                if '\nAnswer the question using a single word or phrase.' not in question:
                    continue
                image_questions.append({
                    'image_path': os.path.join(image_root, os.path.basename(image_path)),
                    'question': question,
                    'answer': answer,
                })
            elif task_type == 'long':
                if '\nAnswer the question using a single word or phrase.' in question:
                    continue
                image_questions.append({
                    'image_path': os.path.join(image_root, os.path.basename(image_path)),
                    'question': question,
                    'answer': answer,
                })
            num += 1
            if max_num is not None and num >= max_num:
                break
        if max_num is not None and num >= max_num:
                break
    random.seed(42)
    image_questions = random.sample(image_questions, extract_num)
    with open(os.path.join(save_dir, f'mem_{task_type}data.json'), 'w') as f:
        json.dump(image_questions, f, indent=4)

def extract_vqav2data(annotation_file_path, question_file_path, img_dir='./data/images/vqav2', num_samples=2000, save_dir='./data/vqav2'):
    with open(annotation_file_path, 'r') as f:
        annotations = json.load(f)
    with open(question_file_path, 'r') as f:
        questions = json.load(f)
    question_map = {q['question_id']: q['question'] for q in questions['questions']}
    
    random.seed(42)
    random.shuffle(annotations['annotations'])
    sampled_annotations = annotations['annotations'][:num_samples]
    
    data = []
    for ann in tqdm(sampled_annotations, desc="Extracting samples"):
        question_id = ann['question_id']
        image_id = ann['image_id']
        question = question_map.get(question_id, "")
        answer = ann['answers'][0]['answer']
        data.append({
            'image_path': f"{img_dir}{str(image_id).zfill(12)}.jpg",
            'question': question + '\nAnswer the question using a single word or phrase.',
            'answer': answer,
        })
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'vqav2_valsamples.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def extract_vizwizdata(val_json_path, img_dir='./data/images/Vizwiz', num_samples=2000, save_dir='./data/Vizwiz'):
    with open(val_json_path, 'r') as f:
        data = json.load(f)
    samples = []

    random.seed(42)
    random.shuffle(data)
    sampled_annotations = data[:num_samples]
    for entry in tqdm(sampled_annotations, desc="Extracting samples"):
        image_path = img_dir + entry['image']
        question = entry['question'] + "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        answer = entry['answers'][0]['answer'] if entry['answers'] else 'no answer'
        sample = {
            'image_path': image_path,
            'question': question,
            'answer': answer
        }
        samples.append(sample)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_sample_path = os.path.join(save_dir, 'vizwiz_valsamples.json')
    with open(output_sample_path, 'w') as f:
        json.dump(samples, f, indent=4)

def extract_ImageNet1Kdata(captions_file, img_dir='./data/images/ImageNet1K_val', num_samples=100, save_dir='./data/ImageNet1K'):
    all_imgs = []
    for img_name in os.listdir(img_dir):
        all_imgs.append(img_name)
    class_dict = {}
    with open(captions_file, 'r') as f:
        for line in f:
            # Skip lines starting with "?" (dummy class)
            if line.startswith('?????????'):
                continue
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_id, class_name = parts
                class_dict[class_id] = class_name.split(',')[0]
    random.seed(42)
    selected_samples = random.sample(all_imgs, num_samples)
    samples = []
    for sample in selected_samples:
        class_id = sample.split('_')[-1].split('.')[0]
        sample_data = {
            "image_path": os.path.join(img_dir, sample),
            "question": "What is the main object in this image?\nAnswer in a single word or phrase.",
            "answer": class_dict[class_id]
        }
        samples.append(sample_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    with open(os.path.join(save_dir, "imagenet1kval_samples.json"), "w") as f:
        json.dump(samples, f, indent=4)

def extract_q_benchdata(json_path, img_dir='./data/images/Q_Bench', save_dir='./data/Q_Bench'):
    samples = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        sample = {
            'image_path': os.path.join(img_dir, entry['img_path']),
            'question': entry['question'] + "\nAnswer the question using a single word or phrase.",
            'answer': entry['correct_ans']
        }
        samples.append(sample)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'qbench_samples.json')
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

def extract_mmedata(json_path, img_dir='./data/images/MME', save_dir='./data/MME'):
    samples = []
    def image_extension(image_dir):
        img_file = os.listdir(image_dir)[0]
        return os.path.splitext(img_file)[1]
    def collect_image(data, img_sourcedir, output_dir):
        for entry in data:
            img_name = os.path.basename(entry['image_path'])
            shutil.copy(os.path.join(img_sourcedir, img_name), os.path.join(output_dir, img_name))
    def extract_from_txt_files(txt_dir, image_dir, img_extent='.jpg'):
        extracted_data = []
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(txt_dir, txt_file)
                image_name = os.path.splitext(txt_file)[0] + img_extent

                with open(txt_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if '\t' in line:
                            question, answer = line.split('\t', 1)
                            extracted_data.append({
                                'image_path': os.path.join(image_dir, image_name),
                                'question': question,
                                'answer': answer
                            })
        return extracted_data
    for task_dir in os.listdir(json_path):
        if len(os.listdir(os.path.join(json_path, task_dir))) == 2:
            img_extent = image_extension(os.path.join(json_path, task_dir, 'images'))
            txt_path = os.path.join(json_path, task_dir, 'questions_answers_YN')
            extracted_data = extract_from_txt_files(txt_path, img_dir, img_extent)
            collect_image(extracted_data, os.path.join(json_path, task_dir, 'images'), img_dir)
        else:
            img_extent = image_extension(os.path.join(json_path, task_dir))
            txt_path = os.path.join(json_path, task_dir)
            extracted_data = extract_from_txt_files(txt_path, img_dir, img_extent)
            collect_image(extracted_data, os.path.join(json_path, task_dir), img_dir)
        samples += extracted_data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'mme_samples.json')
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

def extract_pope_data(json_dir, img_dir='./data/images/POPE', num_samples=2000, save_dir='./data/POPE'):
    samples = []
    for json_file in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        extracted_data = []
        for item in data:
            extracted_sample = {
                'image_path': os.path.join(img_dir, item['image']),
                'question': item['text'] + "\nAnswer the question using a single word or phrase.",
                'answer': item['label']
            }
            extracted_data.append(extracted_sample)
        samples += extracted_data
    random.seed(42)
    selected_samples = random.sample(samples, num_samples)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'pope_samples.json')
    with open(output_path, 'w') as f:
        json.dump(selected_samples, f, indent=4)

def extract_flickr8kdata(captions_file, img_dir='./data/images/Flickr8k', num_samples=100, save_dir='./data/Flickr8k'):
    with open(captions_file, "r") as f:
        lines = f.readlines()
    image_captions = []
    for line in lines:
        img, caption = line.strip().split(",", 1)
        image_captions.append({"image": img, "caption": caption})
    random.seed(42)
    selected_samples = random.sample(image_captions, num_samples)
    samples = []
    for sample in selected_samples:
        img = sample["image"]
        caption = sample["caption"]
        sample_data = {
            "image_path": os.path.join(img_dir, img),
            "question": "Describe the image concisely.",
            "answer": caption
        }
        samples.append(sample_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    with open(os.path.join(save_dir, "flickr8k_samples.json"), "w") as f:
        json.dump(samples, f, indent=4)

def extract_coco2017data(captions_json_path, img_dir='./data/images/COCO2017_val', num_samples=2000, save_dir='./data/COCO2017'):
    with open(captions_json_path, 'r') as f:
        coco_data = json.load(f)
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    samples = []
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        image_name = 'COCO_val2014_' + images.get(image_id)
        if image_name:
            sample = {
                'image_path': os.path.join(img_dir, image_name),
                'question': "Describe the image concisely.",
                'answer': caption,
            }
            samples.append(sample)
    random.seed(42)
    selected_samples = random.sample(samples, num_samples)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'cococaptions_valsamples.json')
    with open(output_path, 'w') as f:
        json.dump(selected_samples, f, indent=4)

def extract_llava_bench_in_wilddata1(val_json_path, img_dir='./data/images/llava_bench_in_wild', save_dir='./data/llava_bench_in_wild'):
    samples = []
    with open(val_json_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            sample = {
                'image_path': os.path.join(img_dir, entry['image']),
                'question': "Describe the image concisely.",
                'answer': entry['caption']
            }
            samples.append(sample)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'llavabenchinthewild_captionsamples.json')
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

def extract_llava_bench_in_wilddata2(question_path, answer_path, img_dir='./data/images/llava_bench_in_wild', save_dir='./data/llava_bench_in_wild'):  
    samples = []
    qts = []
    data = []
    with open(question_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    with open(answer_path, 'r') as f:
        for line in f:
            qt = json.loads(line)
            qts.append(qt)    
    for sample in samples:
        question_text = next((qt['prompt'] for qt in qts if qt['question_id'] == sample["question_id"]), None)
        target_output = next((qt['text'] for qt in qts if qt['question_id'] == sample["question_id"]), None)
        data.append({
            'image_path': os.path.join(img_dir, sample['image']),
            'question': question_text,
            'answer': target_output
        })
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    output_path = os.path.join(save_dir, 'llavabenchinthewild_gpt4samples.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    extract_memdata('your/data/path', task_type='short', data_type=None, max_num=10000, extract_num=2000)
    extract_memdata('your/data/path', task_type='long', data_type=None, max_num=10000, extract_num=2000)
    
    extract_vqav2data('your/data/path', 'your/data/path')
    extract_vizwizdata('your/data/path')
    extract_ImageNet1Kdata('your/data/path')
    extract_q_benchdata('your/data/path')
    extract_mmedata('your/data/path')
    extract_pope_data('your/data/path')
    extract_flickr8kdata('your/data/path')
    extract_coco2017data('.your/data/path')
    extract_llava_bench_in_wilddata1('your/data/path')
    extract_llava_bench_in_wilddata2(
        'your/data/path',
        'your/data/path'
    )
    print("Data extraction completed.")

if __name__ == "__main__":
    main()