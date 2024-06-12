import pickle
from PIL import Image
import requests
import random
from io import BytesIO

import torch
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

    dict = {'url': url, 'question': question, 'caption': caption, 'answers': answers}
    lst_data.append(dict)


with open('/data/train_captions.pkl', 'wb') as f:
    pickle.dump(lst_data, f)