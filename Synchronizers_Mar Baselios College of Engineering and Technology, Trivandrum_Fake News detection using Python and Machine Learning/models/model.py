#training

model= LogisticRegression()
model.fit(x_train, y_train)

#evaluation

#training data aacuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(f'Accuracy of training data : {round(training_data_accuracy*100,2)}%')

#test data accuracy
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f'Accuracy of test data : {round(test_data_accuracy*100,2)}%')