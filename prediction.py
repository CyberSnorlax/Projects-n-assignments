import numpy as np
import csv
import string
import nltk
import collections
import matplotlib.pyplot as plt
from random import uniform, seed

def keyword_in_description(description, keyword):
    return 1 if keyword in description else 0

def ols(X, y):
    X_transpose = X.transpose()
    X_transpose_X = X_transpose.dot(X)
    inv_X_transpose_X = np.linalg.inv(X_transpose_X)
    X_transpose_y = X_transpose.dot(y)
    beta_hat = inv_X_transpose_X.dot(X_transpose_y)
    return beta_hat

def ridge(X, y, fLambda):
    X_transpose = X.transpose()
    X_transpose_X = X_transpose.dot(X)
    lambda_identity = fLambda * np.identity(X_transpose_X.shape[0])
    inv_X_transpose_X_lambda_I = np.linalg.inv(X_transpose_X + lambda_identity)
    X_transpose_y = X_transpose.dot(y)
    beta_hat = inv_X_transpose_X_lambda_I.dot(X_transpose_y)
    return beta_hat

def load_data():
    file = "/Users/williamsempire/Downloads/ps6_package/tables/loan_ridge_small.csv"
    X_raw, y = [], []
    with open(file, "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)
        field_names = dict_reader.fieldnames
        for observation in dict_reader:
            y.append([int(observation["days_until_funded"])])
            observation.pop("days_until_funded", None)
            X_raw.append(observation)
    return X_raw, y, field_names

def preprocessing(X_raw):
    X = []
    word_in_descriptions, countries = [], []
    for observation in X_raw:
        description = observation["description_texts_en"].lower().translate(str.maketrans('', '', string.punctuation))
        word_in_descriptions += description.split()
        countries.append(observation["location_country"].lower())
    word_counter, country_counter = collections.Counter(word_in_descriptions), collections.Counter(countries)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for observation in X_raw:
        x = [float(observation["borrowers_borrower_gender"] == 'M')]
        for word, _ in word_counter.most_common(75):
            if word not in stop_words:
                x.append(keyword_in_description(observation["description_texts_en"], word))
        for country, _ in country_counter.most_common(30):
            x.append(float(observation["location_country"].lower() == country))
        x += [float(observation[k]) for k in ["funded_amount", "terms_disbursal_amount", "repayment_term"]]
        X.append(x)
    return X

def train_test_split(X, y, X_raw, threshold=0.8):
    X_train, X_test, y_train, y_test = [], [], [], []
    X_raw_train, X_raw_test = [], []
    for i, (xi, yi) in enumerate(zip(X, y)):
        if uniform(0, 1) < threshold:
            X_train.append(xi)
            y_train.append(yi)
            X_raw_train.append(X_raw[i])
        else:
            X_test.append(xi)
            y_test.append(yi)
            X_raw_test.append(X_raw[i])
    return X_train, X_test, y_train, y_test, X_raw_train, X_raw_test

def predict(X, beta):
    y_hat = np.dot(X, beta).flatten()
    return y_hat

def mean_squared_error(y_hat, y):
    return np.mean((np.array(y) - y_hat) ** 2)

def recenter(X, y):
    X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
    return X - X_mean, y - y_mean

def save_predictions(X_raw, y_hat, data_label, file_name):
    with open(file_name, mode='a', newline='', encoding="utf8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            headers = list(X_raw[0].keys()) + ["predicted_days_until_funded", "data_set"]
            writer.writerow(headers)
        for obs, pred in zip(X_raw, y_hat):
            row = list(obs.values()) + [pred, data_label]
            writer.writerow(row)

if __name__ == '__main__':
    seed(12409)
    X_raw, y, field_names = load_data()
    X = preprocessing(X_raw)
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(X, y, X_raw, 0.8)

    X_train_matrix = np.matrix(X_train)
    Y_train_matrix = np.array(y_train)
    X_test_matrix = np.matrix(X_test)
    Y_test_matrix = np.array(y_test)

    # # Uncomment this part to center the datapoints. You should start using this part
    # # from Part III, question 2.b
    # X_train_matrix, Y_train_matrix = recenter(X_train_matrix, Y_train_matrix)
    # X_test_matrix, Y_test_matrix = recenter(X_test_matrix, Y_test_matrix)

    # # Use the following line for OLS regression
    # beta = ols(X_train_matrix, Y_train_matrix)
    # print("OLS Beta:", beta)

    # Use the following line for Ridge regression
    beta = ridge(X_train_matrix, Y_train_matrix, 1) #Set the lambda here
    print(f"lambda = 1, beta = {beta}")
    total = sum(number[0] for number in beta)
    print(f"Sum of beta = {total}")
    print(f"Beta average = {np.average(beta)}")

    y_train_pred = predict(X_train_matrix, beta)
    in_sample_mse = mean_squared_error(y_train_pred, Y_train_matrix)
    print("In Sample MSE: " + str(in_sample_mse))

    # Part III 2b)
    X_train, y_train = recenter(np.array(X_train), np.array(y_train))
    X_test, y_test = recenter(np.array(X_test), np.array(y_test))

    beta = ridge(X_train, y_train, 1)
    print(f"for lambda = 1, beta = {beta}")

    X = np.array(preprocessing(X_raw))
    print("Shape of X:", X.shape)

    y_train_pred = predict(X_train, beta)
    y_test_pred = predict(X_test, beta)

    # Part III 3)a)
    # save_predictions(X_raw_train, y_train_pred, "train", "part3a_predictions.csv")
    # save_predictions(X_raw_test, y_test_pred, "test", "part3a_predictions.csv")

    # Part III 3)b)
    print("In Sample MSE:", mean_squared_error(y_train_pred, y_train))
    print("Out Sample MSE:", mean_squared_error(y_test_pred, y_test))

    # Part III 4)
    select_variables = [1, 2, 3, 10, 37, 48, 69]

    lambdas = np.logspace(-2, 2, 30)  # Values from 0.01 to 1000, 30 points
    betas = []
    train_mse = []
    test_mse = []

    for lamb in lambdas:
        beta = ridge(X_train, y_train, lamb)
        betas.append(beta)
        
        y_train_pred = predict(X_train, beta)
        y_test_pred = predict(X_test, beta)
        train_mse.append(mean_squared_error(y_train_pred, y_train))
        test_mse.append(mean_squared_error(y_test_pred, y_test))
    betas = np.array(betas)

    for index in select_variables:
        plt.semilogx(lambdas, betas[:, index], label=f'Beta_{index}')

    # plotting beta for selected variables as lambda increases
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Coefficient Value (Beta)')
    plt.title('Ridge Coefficients vs. Lambda for Selected Variables')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotting MSE vs. Lambda
    plt.semilogx(lambdas, train_mse, label='Training MSE', color='blue')
    plt.semilogx(lambdas, test_mse, label='Test MSE', color='red')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs. Lambda for Training and Test Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print MSE and corresponding lambda
    min_train_mse = min(train_mse)
    min_test_mse = min(test_mse)
    min_train_lambda = lambdas[np.argmin(train_mse)]
    min_test_lambda = lambdas[np.argmin(test_mse)]
    print(f"Minimum Training MSE: {min_train_mse:.4f} at lambda = {min_train_lambda:.4f}")
    print(f"Minimum Test MSE: {min_test_mse:.4f} at lambda = {min_test_lambda:.4f}")