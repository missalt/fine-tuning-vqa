# fine-tuning-vqa


## Documentation

### "How to run everything"

1. If you wish to use the same data set (COCO) you can follow the instructions in this link (https://github.com/missalt/REVIVE-finetuning).
2. make sure you have enough space on your device, approximately 15GB alone for the dataset.
3. mount the data set on your local system or drive.
4. the code itself uses the base model of CLIP. You can use your own preferred version of clip by changing the processor model in the code.
5. if you have no issues with dependencies you should be to run the code and recieve the feature list.
6. if you wish to test your code without iterating through all the images you can simply add a for loop in the Get image URL section of the code and limit the number of added images.   

#### clip_extractor.py
This script is responsible for the Visual Encoding part and generates the features given a set of image url's of the cocodataset with the help of the CLIP model. The final result of the code
is a Pickle file with urls and features paired together.

NOTE! you cannot view the pickle file though normal means. try uncommenting the load pickle file after running the code and print the pickle file if you wish to view its contents.

#### match_and_combine_by_url.py

This script takes the other script's output pickle files as input and merges them together.
The pickle files all share their image URL as a common key. This script uses that key to join the matching questions, answers, answer_candidates and the features for each image.
