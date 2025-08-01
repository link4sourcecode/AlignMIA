import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
# -----------------load target model-----------------
def load_targetmodel(model_path="./checkpoints/LLaVA-7B-v1.5", model_base=None):
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, offload_folder='./offload')
    return model, tokenizer, image_processor, context_len
# -----------------loss and generate function-----------------
def compute_loss(model, tokenizer, image_processor, image_path, targets, device, question="What's the content of the image? ASSISTANT:", 
        conv_mode="llava_v1"):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    if type(image) != torch.Tensor:
        images = process_images([image], image_processor, model.config)[0]
    else:
        images = image
    cur_prompt = question
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    len_prompt_and_question = input_ids.shape[1]
    prompts = {
        "input_ids": [input_ids],
        "context_length": [input_ids.shape[1]],
    }
    context_length = prompts["context_length"]
    context_input_ids = prompts["input_ids"]
    batch_size = len(targets)

    if len(context_input_ids) == 1:
        context_length = context_length * batch_size
        context_input_ids = context_input_ids * batch_size

    images = images.repeat(batch_size, 1, 1, 1)

    assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"
    to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.
    seq_tokens_length = []
    labels = []
    input_ids = []
    for i, item in enumerate(to_regress_tokens):
        L = item.shape[1] + context_length[i]
        seq_tokens_length.append(L)
        context_mask = torch.full([1, context_length[i]], -100,
                                    dtype=to_regress_tokens[0].dtype,
                                    device=to_regress_tokens[0].device)
        labels.append( torch.cat( [context_mask, item], dim=1 ) )
        input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )

    # padding token
    pad = torch.full([1, 1], 0,
                        dtype=to_regress_tokens[0].dtype,
                        device=to_regress_tokens[0].device).cuda()

    max_length = max(seq_tokens_length)

    for i in range(batch_size):
        # padding to align the length
        num_to_pad = max_length - seq_tokens_length[i]
        padding_mask = (
            torch.full([1, num_to_pad], -100,
                    dtype=torch.long,
                    device=device)
        )
        labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )
        input_ids[i] = torch.cat( [input_ids[i],
                                    pad.repeat(1, num_to_pad)], dim=1 )

    labels = torch.cat( labels, dim=0 ).cuda()
    input_ids = torch.cat( input_ids, dim=0 ).cuda()
    with torch.inference_mode():
        outputs = model(
                input_ids=input_ids,
                attention_mask=None,
                return_dict=True,
                labels=labels,
                images=images.half(),
                output_attentions=False,
            )
        loss = outputs.loss
    answer_ids = input_ids[0, len_prompt_and_question:]
    answer_len = len(answer_ids)
    return loss, outputs.logits, answer_len
def generate_output(model, tokenizer, image_processor, image_path, question="What's the content of the image? ASSISTANT:",
               conv_mode="llava_v1", temperature=0.2, top_p=None, num_beams=1, gen_len=50):
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question

    # setup conversation template
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # generate input_ids with image token
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # process image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    if type(image) != torch.Tensor:
        image_tensor = process_images([image], image_processor, model.config)[0]
    else:
        image_tensor = image

    # generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            # do_sample=True if temperature > 0 else False,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=gen_len,
            use_cache=True
        )
    # decode output_ids to text
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs
# -----------------image processing-----------------
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
def process_images(images, image_processor, model_cfg, pad_idx, patch_size=14):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = Image.open(image).convert('RGB')
            padding_image = Image.new(image.mode, image.size, tuple(int(x*255) for x in image_processor.image_mean))
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            padding_image = expand2square(padding_image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            padding_image = image_processor.preprocess(padding_image, return_tensors='pt')['pixel_values'][0]
            for idx in pad_idx:
                row = idx // (image.shape[1] // patch_size)
                col = idx % (image.shape[1] // patch_size)
                image[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = padding_image[:, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images