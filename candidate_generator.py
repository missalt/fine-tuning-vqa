import pickle
from PIL import Image

import requests
import random
from io import BytesIO
import torch
import transformers

# Parameter for beam search
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


def generate_answercandidates(file_path, output_path):
    """
    Function to generate answer candidates for images.

    Parameters:
    - file_path (str): Path to the input pickle file containing image data.
    - output_path (str): Path to the output pickle file to save results.

    The function loads the captions and corresponding questions from the input file and generates answer candidates. 
    The results are saved to the specified output file.
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    lst_data = []
    for i, dict_data in enumerate(data):
        question = dict_data['question']
        caption = dict_data['caption']
        url = dict_data['url']

        if i % 100 == 0:
            print(f"Processing caption {i} of {len(data)}")

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


        dict_new = {'url': url, 'question': question, 'caption': caption, 'answers': answer_candidates}
        lst_data.append(dict_new)


    with open(output_path, 'wb') as f:
        pickle.dump(lst_data, f)


# Generate captions and answer candidates for training and testing datasets
generate_answercandidates('out_data/train_caption.pkl', 'out_data/train_answers_captions.pkl')
generate_answercandidates('out_data/test_caption.pkl', 'out_data/test_answers_captions.pkl')