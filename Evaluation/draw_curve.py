import numpy as np
import matplotlib.pyplot as plt

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times

if __name__ == "__main__":

    train_loss_path = "eval_loss.txt"
    train_acc_path = "eval_acc.txt"

    y_train_loss = data_read(train_loss_path)
    y_train_acc = data_read(train_acc_path)

    x_train_loss = range(len(y_train_acc))
    x_train_acc = x_train_loss

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('accuracy')

    plt.plot(x_train_loss, y_train_loss,  color='red', linestyle="solid", label="eval loss")
    plt.legend()

    plt.title('acc curve')
    plt.savefig('./eval_loss.jpg')
    plt.show()