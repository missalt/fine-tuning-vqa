import pickle
from PIL import Image
import requests
import random
from io import BytesIO

import torch
import transformers
from promptcap import PromptCap

model = PromptCap("tifa-benchmark/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"

# device = "cuda:0"
# model.to(device)
model = model.cuda()

with open('REVIVE/processed_data/train.pkl', 'rb') as f:
    data = pickle.load(f)

processed_imgs = []

for i, dict_data in enumerate(data):
    question = dict_data['question']
    img_id = dict_data['image_name']
    answers = dict_data['answers_list']
    split = img_id.split('_')[1]
    url = f"http://images.cocodataset.org/{split}/{img_id}"
    processed_imgs.append((url, question))

lst_data = []
for i, (url, question) in enumerate(processed_imgs):
    if i % 100 == 0:
        print(f"Processing image {i} of {len(processed_imgs)}")
    prompt = f"please describe this image according to the given question: {question}"
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    caption = model.caption(prompt, image_bytes)

    ########################################
    # Generating the answer candidates

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda"
    )

    answer_candidates = []
    n = 10

    messages = [
        {"role": "system", "content": f"context: {caption}, question: {question}, answer the question in max 2 words and no punctuation"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )


    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate answer candidates using beam search
    outputs = pipeline(
        prompt,
        max_new_tokens=10,
        eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
        num_beams=n,
        num_return_sequences=n,
    )

    answer_candidates = [output["generated_text"][len(prompt):].strip() for output in outputs]

    ########################################

    dict = {'url': url, 'question': question, 'caption': caption, 'answers': answer_candidates}
    lst_data.append(dict)


with open('/data/train_captions.pkl', 'wb') as f:
    pickle.dump(lst_data, f)