from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np 
from fitting_model import *

def train(all_targets, all_states, real_targets, train_points = 395, delay_points = 5, fold_nums=[0,1,2,3,4], num_ft=36):
    train_points = train_points
    delay_points = delay_points
    observation = train_points + delay_points
    results = {}
    errors = []
    digits = [int(i) for i in range(10)]

    for fold_num in fold_nums: 
        ### Combine STD and Mean of each fold together for computation
        name_state = str(fold_num)+'state_mean'+str(num_ft)
        states_mean = all_states[name_state]
        name_state = str(fold_num)+'state_std'+str(num_ft)
        states_std = all_states[name_state]
        states = np.concatenate((states_mean, states_std), axis=1) # combine std and mean
    
        # extract train and test features
        X_train = states[delay_points:train_points+delay_points]
        X_test = states[train_points+delay_points:]
        for digit in digits: # for each digit
            name_target = str(fold_num) + '_digit_' + str(digit) +'_'+ str(num_ft) #NOTE: target of 36 is the same as 49. 
            target = all_targets[name_target] # get the target in 0,1 
            y_train = target[delay_points:train_points+delay_points] # separate into train and test
            y_test = target[train_points+delay_points:]
            W = fitting_function(X_train, y_train) # find the weight matrix
            y_prediction = predict(X_test, W) # find the final prediction
            results[name_target] = y_prediction 
            # results store all predictions for ten classifiers 
    
        # Convert results into actual labels
        prediction = []
        label = -1
        for i in range(len(y_prediction)):
            max_value = -10
            for value in results:
                label = value.split('_')[2] # classifier label
                fold = value.split('_')[0] # fold number
                fts = value.split('_')[3] # amount of features
                if results[value][i] > max_value and int(fold)==fold_num: 
                    max_value = results[value][i]
                    predicted_label = int(label)
            prediction.append(predicted_label)
        #print("The predictions are: ", (prediction))
        real_answer = real_targets[str(fold_num)+'_'+str(num_ft)][observation:]
        #print("The actual labels are: ", real_answer)
    
        # Convert to WER
        substitution = 0 
        for i in range(len(real_answer)):
            if real_answer[i] != prediction[i]:
                substitution += 1
        WER = substitution/len(real_answer)
        errors.append(WER)
        #print("WER: " + str(WER))
    print(f" ",errors)
    return errors