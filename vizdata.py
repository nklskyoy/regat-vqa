import json
import matplotlib.pyplot as plt
import pickle
from PIL import Image

def compute_accuracy(ground_truth_file, predicted_answers_file):
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    with open(predicted_answers_file, 'r') as f:
        predicted_answers_data = json.load(f)


    ground_truth_dict = {entry['question_id']: entry['multiple_choice_answer'] for entry in ground_truth_data['annotations']}
    predicted_dict = {entry['question_id']: entry['answer'] for entry in predicted_answers_data}

    correct_count = 0
    total_count = len(ground_truth_dict)

    for question_id, ground_truth_answer in ground_truth_dict.items():
        predicted_answer = predicted_dict.get(question_id, '')
        if ground_truth_answer == predicted_answer:
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    return accuracy

    # Code to compute accuracy

def visualize_answers(ground_truth_file, predicted_answers_file,question_file,pickle_file,image_file):
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    with open(predicted_answers_file, 'r') as f:
        predicted_answers_data = json.load(f)

    with open(question_file, 'r') as f:
        question_data = json.load(f)

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
        

    with open(image_file, 'r') as f:
        image_data = json.load(f)

    ground_truth_dict = {entry['question_id']: entry['multiple_choice_answer'] for entry in ground_truth_data['annotations']}
    predicted_dict = {entry['question_id']: entry['answer'] for entry in predicted_answers_data}
    question_dict = {entry['question_id']: entry['question'] for entry in question_data['questions']}
    image_dict={entry['id']: entry['file_name'] for entry in image_data['images']}
    

    count=0

    for question_id, predicted_answer in predicted_dict.items():
        if question_id in ground_truth_dict:
            ground_truth_entry = ground_truth_dict[question_id]
            print(question_id)
            question=question_dict.get(question_id)
            for item in question_data['questions']:
                if item["question_id"] == question_id:
                    image_id = item["image_id"]
            image_name=image_dict.get(image_id)
            image_path = f"/home/jm351565/fiftyone/coco-2014/validation/data/{image_name}"
            image = Image.open(image_path)
            image.show()
            #question = next((q for q, qid in question_dict.items() if qid == question_id),None)
            ground_truth_answer = ground_truth_entry

            print(f"Question: {question}")
            print(f"Ground Truth Answer: {ground_truth_answer}")
            print(f"Predicted Answer: {predicted_answer}")
            print("-----------------------")


            count += 1
            if count == 1:
                break  # Stop after the first 10 entries
              
            
      
      



# Usage example
ground_truth_file = '/hpcwork/lect0099/data/Answers/v2_mscoco_val2014_annotations.json'
predicted_answers_file = '/hpcwork/lect0099/saved_models/gl671475/bs-lr/ban_1_spatial_vqa_200_bs_64_lr_0.01_ep_20/eval/vqa_val.json'
question_file='/hpcwork/lect0099/data/Questions/v2_OpenEnded_mscoco_val2014_questions.json'
pickle_file='/hpcwork/lect0099/data/cache/val_target.pkl'
image_file='/home/jm351565/fiftyone/coco-2014/validation/labels.json'

accuracy = compute_accuracy(ground_truth_file, predicted_answers_file)
print(f"Accuracy: {accuracy}%")

visualize_answers(ground_truth_file, predicted_answers_file,question_file,pickle_file,image_file)