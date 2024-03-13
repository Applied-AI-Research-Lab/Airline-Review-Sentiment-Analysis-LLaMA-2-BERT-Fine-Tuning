import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns


class EvaluationMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/'

    def evaluate_results(self, original, prediction, model_name):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        accuracy = round(accuracy_score(data[original], data[prediction]), 4)
        precision = round(precision_score(data[original], data[prediction], average='weighted'), 4)
        recall = round(recall_score(data[original], data[prediction], average='weighted'), 4)
        f1 = round(f1_score(data[original], data[prediction], average='weighted'), 4)

        # Create a DataFrame with the evaluation results including the 'model' column
        evaluation_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1]
        })

        # Append the results to the existing CSV file or create a new one
        evaluation_df.to_csv(self.pre_path + 'evaluation-results.csv', mode='a',
                             header=not os.path.exists(self.pre_path + 'evaluation-results.csv'), index=False)

        return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    def scatterplot(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)
        prediction = df[prediction_column]
        original = df[original_column]

        # Create a scatter plot with a regression line
        sns.regplot(x=original, y=prediction, scatter_kws={'alpha': 0.5})

        plt.xlabel(original_column)
        plt.ylabel(prediction_column)

        # Save the scatterplot image to the Datasets folder
        plt.savefig(os.path.join(self.pre_path + 'Plots/', prediction_column + '.png'))

        # Show the plot
        plt.show()


# Example Usage
# Instantiate the DatasetMethods class by providing the (dataset_path)
EVM = EvaluationMethods(dataset_path='test_set.csv')

# # Evaluate the predictions made by each model
# print(f'base:llama-2-7b-chat: ' + str(EVM.evaluate_results('Overall_Rating', 'llama_bm_prediction', 'base:llama-2-7b-chat')))
# print(f'ft-2k:llama-2-7b-chat: ' + str(EVM.evaluate_results('Overall_Rating', 'llama_ft_2k_prediction', 'ft-2k:llama-2-7b-chat')))
# print(f'ft-6k:llama-2-7b-chat: ' + str(EVM.evaluate_results('Overall_Rating', 'llama_ft_6k_prediction', 'ft-6k:llama-2-7b-chat')))
# print(f'ft-2k:bert-adamw: ' + str(EVM.evaluate_results('Overall_Rating', 'bert_adamw_ft_2k_prediction', 'ft-2k:bert-adamw')))
# print(f'ft-2k:bert-adam: ' + str(EVM.evaluate_results('Overall_Rating', 'bert_adam_ft_2k_prediction', 'ft-2k:bert-adam')))
# print(f'ft-6k:bert-adamw: ' + str(EVM.evaluate_results('Overall_Rating', 'bert_adamw_ft_6k_prediction', 'ft-6k:bert-adamw')))
# print(f'ft-6k:bert-adam: ' + str(EVM.evaluate_results('Overall_Rating', 'bert_adam_ft_6k_prediction', 'ft-6k:bert-adam')))

# Create scatterplots
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='llama_bm_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='llama_ft_2k_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='llama_ft_6k_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='bert_adamw_ft_2k_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='bert_adam_ft_2k_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='bert_adamw_ft_6k_prediction')
# EVM.scatterplot(original_column='Overall_Rating', prediction_column='bert_adam_ft_6k_prediction')
