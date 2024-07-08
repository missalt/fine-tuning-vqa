import pickle
from PIL import Image
import requests
import random
from io import BytesIO
import torch
from promptcap import PromptCap

# Initialize the PromptCap model
model = PromptCap("tifa-benchmark/promptcap-coco-vqa")

# device = "cuda:0"
# model.to(device)
model = model.cuda()

def generate_caption(file_path, output_path):
    """
    Function to generate captions and answer candidates for images.

    Parameters:
    - file_path (str): Path to the input pickle file containing image data.
    - output_path (str): Path to the output pickle file to save results.

    The function loads image data from the input file and generates captions using the PromptCap model.
    The results are saved to the specified output file.
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    processed_imgs = []

    for i, dict_data in enumerate(data):
        question = dict_data['question']
        img_id = dict_data['image_name']
        split = img_id.split('_')[1]
        url = f"http://images.cocodataset.org/{split}/{img_id}"
        processed_imgs.append((url, question))

    lst_data = []
    for i, (url, question) in enumerate(processed_imgs):
        if i % 100 == 0:
            print(f"Processing image {i} of {len(processed_imgs)}")

        # Generating the caption
        prompt = f"please describe this image according to the given question: {question}"
        
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # Generate caption using the PromptCap model
        caption = model.caption(prompt, image_bytes)

        dict = {'url': url, 'question': question, 'caption': caption,}
        lst_data.append(dict)


    with open(output_path, 'wb') as f:
        pickle.dump(lst_data, f)


# Generate captions and answer candidates for training and testing datasets
generate_caption('processed_data/train_output.pkl', 'out_data/train_captions.pkl')
generate_caption('processed_data/test_output.pkl', 'out_data/test_captions.pkl')