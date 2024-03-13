import os
import pandas as pd
import replicate
import json
import logging


class LLaMAmethods:
    def __init__(self,
                 model_id="meta/llama-2-7b-chat", prediction_type='bm'):
        self.model_id = model_id
        self.pre_path = 'Datasets/'
        self.prediction_type = prediction_type

    """
    Create a training and validation JSONL file for LLaMA-2 fine-tuning
    """

    def create_jsonl(self, data_type, data_set):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = []  # Define a list to store the dictionaries

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            data.append(
                {
                    "prompt": 'As an airline passenger, you are required to rate your experience as a customer. ' +
                              'Predict the star rating (integer between 1 and 9) to the following review.'
                              'Return your response in json format like this example {"rating":integer}.'
                              'Please avoid providing additional explanations. Review:\n' +
                              row['Review_Title'] + ' ' + row['Review'] + '',
                    "completion": '{"rating":' + str(row['Overall_Rating']) + '}'
                }
            )  # TODO! Change it!

        output_file_path = self.pre_path + "ft_dataset_llama_" + data_type + ".jsonl"  # Define the path
        # Write data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                # Convert the dictionary to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    """
    Create a conversation with LLaMA-2 model
    """

    def llama_conversation(self, conversation):
        result_string = ""
        output = replicate.run(
            self.model_id,
            input=conversation
        )

        # The predict method returns an iterator, and you can iterate over that output.
        for item in output:
            result_string += str(item)

        return result_string

    # You can pass extra parameters in input except prompt
    # input = {
    #     "debug": False,
    #     "top_k": -1,
    #     "top_p": 1,
    #     "prompt": "Tell me how to tailor a men's suit so I look fashionable.",
    #     "temperature": 0.75,
    #     "system_prompt": "You are a helpful, respectful and honest assistant.",
    #     "max_new_tokens": 800,
    #     "min_new_tokens": -1,
    #     "repetition_penalty": 1
    # }

    # Old function
    # def llama_conversation(self, conversation):
    #     result_string = ""
    #     for event in replicate.stream(
    #             self.model_id,
    #             input=conversation
    #     ):
    #         result_string += str(event)
    #
    #     return result_string

    """
    Clean the response
    """

    def clean_response(self, response, a_field):
        # Search for JSON in the response
        start_index = response.find('{')
        end_index = response.rfind('}')

        if start_index != -1 and end_index != -1:
            json_str = response[start_index:end_index + 1]
            try:
                # Attempt to load the extracted JSON string
                json_data = json.loads(json_str)
                return {"status": True, "data": json_data}
            except json.JSONDecodeError as e:
                # If an error occurs during JSON parsing, handle it
                logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                              f"resulted in the following response: {response}")
                return {"status": False,
                        "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                                f"resulted in the following response: {response}"}
        else:
            logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
                          f"following response: {response}")
            return {"status": False, "data": f"No JSON found in the response. The input '{a_field}', "
                                             f"resulted in the following response: {response}"}

    """
    Prompt the LLaMA model to make a prediction
    """

    def llama_prediction(self, input):
        conversation = {'prompt': 'As an airline passenger, you are required to rate your experience as a customer. '
                                  'Predict the star rating (integer between 1 and 9) to the following review.'
                                  'Return your response in json format like this example {"rating":integer}.'
                                  'Please avoid providing additional explanations. Review:\n' +
                                  input['Review_Title'] + ' ' + input['Review'] + ''}  # TODO! Change it!
        content = self.llama_conversation(conversation)  # Get the response from LLaMA model

        # Clean the response and return
        return self.clean_response(response=content, a_field=input['Review_Title'])  # TODO! Change it!

    """
    Make predictions for a specific data_set appending a new prediction_column
    """

    def predictions(self, data_set, prediction_column):

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + data_set)

        # make a copy to _original1
        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if prediction_column not in df.columns:  # TODO! For non-int values omit this if
            # If not, add the column to the DataFrame with pd.NA as the initial value
            df[prediction_column] = pd.NA

            # Explicitly set the column type to a nullable integer
            df = df.astype({prediction_column: 'Int64'})

        # Update the CSV file with the new header (if columns were added)
        if prediction_column not in df.columns:
            df.to_csv(self.pre_path + data_set, index=False)

        # Set the dtype of the reason column to object
        # df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # If the prediction column is NaN then proceed to predictions
            if pd.isnull(row[prediction_column]):
                prediction = self.llama_prediction(input=row)
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)

                    if not prediction['data']['rating']:  # TODO! Change it!
                        logging.error(
                            f"No rating instance was found within the data for '{row['Review_Title']}', and the "
                            f"corresponding prediction response was: {prediction}.")  # TODO! Change it!
                        return {"status": False,
                                "data": f"No rating instance was found within the data for '{row['Review_Title']}', "
                                        f"and the corresponding prediction response was: {prediction}."}  # TODO! Change it!
                    else:
                        # Update the DataFrame with the evaluation result
                        df.at[index, prediction_column] = int(prediction['data']['rating'])  # TODO! Change it!

                        # Update the CSV file with the new evaluation values
                        df.to_csv(self.pre_path + data_set, index=False)

                # break
            # Add a delay of 5 seconds (reduced for testing)

        # Change the column datatype after processing all predictions to handle 2.0 ratings
        df[prediction_column] = df[prediction_column].astype('Int64')  # TODO! For non-int values omit this if

        return {"status": True, "data": 'Prediction have successfully been'}

    """
    Train LLaMA-2 model
    """

    def llama_train(self, destination, train_data, validation_data):
        # https://replicate.com/blog/fine-tune-llama-2
        # https://replicate.com/docs/guides/fine-tune-a-language-model
        training = replicate.trainings.create(
            version=self.model_id,
            input={
                "train_data": train_data,  # The URL of the JSONL file uploaded to a live web server
                "validation_data": validation_data,  # The URL of the JSONL file uploaded to a live web server
                "num_train_epochs": 3,  # Number of epochs (iterations over the entire training dataset) to train for.
                # You can use more training parameters https://replicate.com/meta/llama-2-7b-chat/train
            },
            destination=destination  # The model created through the Replicate U
        )

        print(training)

    def llama_list_train(self):
        training = replicate.trainings.list()
        print(training)

    def llama_check_status(self, training_id):
        training = replicate.trainings.get(training_id)
        print(training)
        if training.status == "succeeded":
            print(training.output)
            # {"weights": "...", "version": "..."}

    def llama_cancel_trainings(self, training_id):
        try:
            # Cancel the training
            training = replicate.trainings.cancel(training_id)
            print(training)
        except replicate.exceptions.ReplicateError as e:
            print(f"Error cancelling training: {e}")


# Example Usage
# Configure logging to write to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# # Instantiate the LLaMAmethods class
# LLaMA = LLaMAmethods()
#
# # Create the json file for training
# LLaMA.create_jsonl(data_type='train', data_set='train_set.csv')  # You have to change the prompt text on each project
#
# # Create the json file for validation
# LLaMA.create_jsonl(data_type='validation', data_set='validation_set.csv')

# Train LLaMA-2 model
# LLaMA = LLaMAmethods(model_id='meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0')
# LLaMA.llama_train(destination='kroumeliotis/airline-reviews',  # The model created through the Replicate UI
#                   train_data='https://acloud.gr/ai/airlines/ft_dataset_llama_train.jsonl',
#                   validation_data='https://acloud.gr/ai/airlines/ft_dataset_llama_validation.jsonl')
# id='i3rcjxlb7qy76utmggers5nqca' model='meta/llama-2-7b-chat' version='13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0' destination=None status='starting' input={'num_train_epochs': 3, 'train_data': 'https://acloud.gr/ai/airlines/ft_dataset_llama_train.jsonl', 'validation_data': 'https://acloud.gr/ai/airlines/ft_dataset_llama_validation.jsonl'} output=None logs='' error=None created_at='2024-03-11T08:47:31.511610941Z' started_at=None completed_at=None urls={'cancel': 'https://api.replicate.com/v1/predictions/i3rcjxlb7qy76utmggers5nqca/cancel', 'get': 'https://api.replicate.com/v1/predictions/i3rcjxlb7qy76utmggers5nqca'}

# LLaMA.llama_check_status(training_id='mnsdgklbqq3qbzu4zj7fmlgula')  # Check training status and get training result data
# LLaMA.llama_list_train()  # Get all trained models
# LLaMA.llama_cancel_trainings(training_id='mnsdgklbqq3qbzu4zj7fmlgula')  # Cancel training process

# Make predictions before Fine-tuning using the Base Model
# LLaMA = LLaMAmethods(model_id='meta/llama-2-7b-chat')
# LLaMA.predictions(data_set='test_set.csv', prediction_column='llama_bm_prediction')

# Make predictions after Fine-tuning using the Fine-tuned (model_id)
# LLaMA = LLaMAmethods(
#     model_id='kroumeliotis/airline-reviews:74541a7a500c2c794e0cf034b562aadf8b60b1a882aa56efe87accc3f5e7691a')
# LLaMA.predictions(data_set='test_set.csv', prediction_column='llama_ft_prediction')
