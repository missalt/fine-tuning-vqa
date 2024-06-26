# fine-tuning-vqa


## Documentation

### "How to run everything"

1. Run the clip_extractor.py and the ... .py script to generate the seperate pickle files containing the features and the question/answer pairs.
2. Run the match_and_combine_by_url.py script to merge them. The output file is specified in the PATH_OUTPUT Variable and will be /data/train_output.pkl by default.

#### clip_extractor.py
This script is responsible for the Visual Encoding part and generates the features given a set of image url's of the cocodataset with the help of the CLIP model.

#### match_and_combine_by_url.py

This script takes the other script's output pickle files as input and merges them together.
The pickle files all share their image URL as a common key. This script uses that key to join the matching questions, answers, answer_candidates and the features for each image.
