import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.registry import registry
from minigpt4.common.utils import now
from minigpt4.common.logger import setup_logger
from minigpt4.common.config import Config
from types import SimpleNamespace
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from PIL import Image
from transformers import StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub
# -----------------load target model-----------------
def setup_seeds():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls

def load_targetmodel(cfg_path):
    job_id = now()
    gpu_id = 0
    args = SimpleNamespace(
    cfg_path=cfg_path,
    gpu_id=gpu_id,
    options=None
    )
    cfg = Config(args)
    setup_seeds()
    # set after init_distributed_mode() to only log on master.
    setup_logger()
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    return model, None, vis_processor, None
# -----------------loss and generate function-----------------
def compute_loss(model, tokenizer, image_processor, image_path, answer, device, question="What's the content of the image? ASSISTANT:", conv_mode=None):
    if type(image_path) == torch.Tensor:
        image = image_path.to(device)
    else:
        image = Image.open(image_path).convert("RGB")
        image = image_processor(image).unsqueeze(0).to(device)
    if "<Img><ImageHere></Img>" in question:
        instruction_input = question
    else:
        instruction_input = '<Img><ImageHere></Img> ' + question
    ann = {
        "image": image,
        "answer": answer,
        "instruction_input": instruction_input,
    }
    with torch.inference_mode():
        outputs = model(ann)
    answer_ids = [idx for idx, id in enumerate (outputs['targets'][0]) if id != -100]
    return outputs['loss'], outputs['logits'], len(answer_ids)

# Also you can add the chat template here
def generate_output(model, tokenizer, vis_processor, image_path, question, conv_mode=None, 
                     temperature=1.0, top_p=0.9, num_beams=1, gen_len=200):
    if type(image_path) == torch.Tensor:
        image = image_path.cuda()
    else:
        image = Image.open(image_path).convert("RGB")
        image = vis_processor(image).unsqueeze(0).cuda()
    if "<Img><ImageHere></Img>" in question:
        question = question
    else:
        question = '<Img><ImageHere></Img> ' + question
    img_emb = model.encode_img(image)[0]
    embs = model.get_context_emb(question, [img_emb])

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'cuda:0') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    with model.maybe_autocast():
        output_token = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=gen_len,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_length=1,
            top_p=top_p,
            repetition_penalty=1,
            length_penalty=1,
            temperature=temperature,
        )[0]
    output_text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text

# -----------------image processing-----------------
def process_images(images, image_processor, model_cfg, pad_idx, patch_size=14):
    new_images = []
    for image in images:
        image = Image.open(image).convert('RGB')
        image = image_processor(image)
        padding_image = Image.new('RGB', (224, 224), tuple(int(x*255) for x in [0.48145466, 0.4578275, 0.40821073]))
        padding_image = image_processor(padding_image)
        for idx in pad_idx:
            row = idx // (image.shape[1] // patch_size)
            col = idx % (image.shape[1] // patch_size)
            image[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = padding_image[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
        new_images.append(image)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return (new_images,)