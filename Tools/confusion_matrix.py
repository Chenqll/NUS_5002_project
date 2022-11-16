
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot(matrix):
        matrix = matrix
        from sklearn.preprocessing import MinMaxScaler
        scalar = MinMaxScaler(feature_range=(0, 1))  
        matrix = scalar.fit_transform(matrix)  

        # matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print(matrix)


        plt.imshow(matrix, cmap=plt.cm.Paired)

        # 设置x轴坐标label
        plt.xticks(range(3), [0,1,2], rotation=85)
        # 设置y轴坐标label
        plt.yticks(range(3), [0,1,2])
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        # for x in range(3):
        #     for y in range(3):
        #         # 注意这里的matrix[y, x]不是matri                                                                                                                                                                        x[x, y]
        #         info = "%.2f" % matrix[y, x]
        #         print(info)
        #         plt.text(x, y, info,
        #                  verticalalignment='center',
        #                  horizontalalignment='center',
        #                  )
        
        
        plt.tight_layout()
        # plt.text(matrix)
        plt.savefig('./confusion_matrics_LR.jpg')
        plt.show()

A=[[ 421 ,120 , 173],
    [  88  ,948 , 381],
    [  25  ,155 ,3277]]

# [[1.         0.         0.        ]
#  [0.15909091 1.         0.06701031]
#  [0.         0.04227053 1.        ]]

B=[[ 448 , 112   ,38],
 [ 105  ,727 , 298],
 [  31  ,164 ,2679]]

# [[1.         0.         0.        ]
#  [0.17745803 1.         0.09844756]
#  [0.         0.08455285 1.        ]]


C=[[ 272 , 205  ,108],
 [ 123  ,813  ,223],
 [  84  ,369 ,2405]]

# [[1.         0.         0.        ]
#  [0.20744681 1.         0.0500653 ]
#  [0.         0.26973684 1.        ]]

plot(C)