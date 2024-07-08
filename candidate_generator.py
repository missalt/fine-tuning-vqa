import pickle
from PIL import Image
import requests
import random
from io import BytesIO

import torch
import transformers
from promptcap import PromptCap

# Initialize the PromptCap model
model = PromptCap("tifa-benchmark/promptcap-coco-vqa")
n = 10

# Initialize the text generation pipeline with Meta-Llama model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda"
)

# Define terminators for the pipeline
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]



# device = "cuda:0"
# model.to(device)
model = model.cuda()

def generate_caption_answercandidates(file_path, output_path):
    """
    Function to generate captions and answer candidates for images.

    Parameters:
    - file_path (str): Path to the input pickle file containing image data.
    - output_path (str): Path to the output pickle file to save results.

    The function loads image data from the input file, generates captions using the PromptCap model,
    and generates answer candidates using the text generation pipeline. The results are saved to the
    specified output file.
    """

    with open(file_path, 'rb') as f:
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

        ########################################
        # Generating the caption
        prompt = f"please describe this image according to the given question: {question}"
        
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # Generate caption using the PromptCap model
        caption = model.caption(prompt, image_bytes)

        ########################################
        # Generating the answer candidates
        answer_candidates = []

        messages = [
            {"role": "system", "content": f"context: {caption}, question: {question}, answer the question in max 2 words and no punctuation"},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        # Generate multiple answer candidates
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

        # Extract the generated answers from the outputs
        answer_candidates = [output["generated_text"][len(prompt):].strip() for output in outputs]

        ########################################

        dict = {'url': url, 'question': question, 'caption': caption, 'answers': answer_candidates}
        lst_data.append(dict)


    with open(output_path, 'wb') as f:
        pickle.dump(lst_data, f)


# Generate captions and answer candidates for training and testing datasets
generate_caption_answercandidates('processed_data/train_output.pkl', '/out_data/train_answers_captions.pkl')
generate_caption_answercandidates('processed_data/test_output.pkl', '/out_data/test_answers_captions.pkl')