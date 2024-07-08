import pickle

PATH_CAPTIONS_ANSWERS = "out_data/train_answers_captions.pkl"
PATH_FEATURES = "out_data/output_features.pkl"

PATH_OUTPUT = "out_data/train_output.pkl"

def match_and_combine_by_url():

    output = []
    with open(PATH_CAPTIONS_ANSWERS, 'rb') as f_train_answers:
        with open(PATH_FEATURES, 'rb') as f_output_features:
            data_train_answers = pickle.load(f_train_answers)
            data_output_features = pickle.load(f_output_features)
            for i in range(len(data_train_answers)):
                if data_train_answers[i]["url"] == data_output_features[i]["url"]:
                    output.append({'url': data_train_answers[i]["url"], 
                                   'question': data_train_answers[i]['question'], 
                                   'caption': data_train_answers[i]['caption'], 
                                   'answers': data_train_answers[i]['answers'], 
                                   'answer_candidates': data_train_answers[i]['answer_candidates'], 
                                   'feature': data_output_features[i]['feature']
                                   })
                else:
                    print("Error: URLs dont match") # This shouldn't happen as the url should always be the same for all arrays at i
    return output


def save_array_to_pickle(array, file_path):
    # Save the array to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(array, file)


if __name__ == '__main__':

    save_array_to_pickle(match_and_combine_by_url(), PATH_OUTPUT)
