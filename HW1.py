import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Ref.
    Gradient Descent: https://reurl.cc/X4Mpba
    Linear regression plot: https://reurl.cc/akmz3G
"""


def load_csv(csv_path, mode='train'):
    df = pd.read_csv(csv_path)
    x_key, y_key = f'x_{mode}', f'y_{mode}'
    x_tmp, y_tmp = df[x_key], df[y_key]
    # Reshape the data
    x = np.array(x_tmp).reshape(-1, 1)
    y = np.array(y_tmp).reshape(-1, 1)

    return x, y


def MSE(diff):
    return (diff ** 2).sum() / diff.shape[0]


def derivative(X_batch, y_batch, theta):
    y_pred = X_batch @ theta
    n = len(X_batch)

    df_dm = (-2/n) * (X_batch.T @ (y_batch - y_pred))

    return df_dm, y_batch - y_pred


def GD(x, y, Lr=1e-4, batch=None, iter=200):

    # random initial　weights and intercepts
    theta = np.random.randn(x.shape[1], 1)
    print('Init. weight and intercepts: ', theta[1][0], ", ", theta[0][0])
    loss_record = []

    for i in range(iter):
        idx = np.random.randint(0, x.shape[0], batch)
        x_batch = x[idx]
        y_batch = y[idx]

        df_dm, diff = derivative(x_batch, y_batch, theta)
        theta -= Lr * df_dm
        loss_record.append(MSE(diff))

        # print(f"Iteration: {i} ----> MSE: {MSE(diff)}")

    return theta, loss_record


def set_hyperparameter(mode, x):
    print('================')
    if mode == 1:
        name = 'Gradient Descent'
        lr = 1e-1
        batch = x.shape[0]  # training data size
    elif mode == 2:
        name = 'Mini-Batch Gradient Descent'
        lr = 1e-1
        batch = 20
    else:
        name = 'Stochastic Gradient Descent'
        lr = 1e-2
        batch = 1
    print(name)

    return lr, batch, name


# Load training data
x_train, y_train = load_csv('train_data.csv', 'train')

# Extend trainging data for calculating
x_train_re = np.concatenate((np.ones(x_train.shape), x_train), axis=1)

# print(x_train_re)
print('Please input the number for choosing which type:\n',
      '(1) Gradient Descent\n (2) Mini-Batch Gradient Descent\n',
      '(3) Stochastic Gradient Descent')
GDtype = input()
GDtype = int(GDtype)

lr, batch, name = set_hyperparameter(GDtype, x_train_re)
theta, mse = GD(x_train_re, y_train,
                Lr=lr, batch=batch)

print('After training, weight: ', theta[1][0],
      ' and intercepts: ', theta[0][0])

# Testing and calculate MSE
x_test, y_test = load_csv('test_data.csv', 'test')
x_test = np.concatenate((np.ones(x_test.shape), x_test), axis=1)
y_pred = x_test @ theta
error = MSE(y_test - y_pred)
print('Mean square error: ', error)


# Draw Linear regression line and loss
x = x_train.flatten()
y = y_train.flatten()
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)

plt.subplot(211)
plt.title(name)
plt.scatter(x, y, s=5, c="#278AA8")
plt.plot(x, poly1d_fn(x), c="#C9306B")

plt.subplot(212)
plt.title("Loss")
plt.scatter(np.arange(len(mse)), mse, s=5, c="#327185")

plt.tight_layout()
plt.savefig(name, dpi=150)
plt.show()
