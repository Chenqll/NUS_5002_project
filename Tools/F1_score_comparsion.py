
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cv_res_value=[0.78,0.78,0.657434,0.626781,0.614093,0.549255,0.509188,0.485326,0.447356,0.186031]
cv_res_keys=['bert','xlnet','lr','lgbm','xgb','svm','rf','nb','gbt','knn']

results1 = pd.DataFrame(cv_res_value, index=cv_res_keys)

results1.sort_values(by=[0], inplace=True, ascending=True)

results1.plot(kind="barh")

# print(results1.index)

plt.title('Comparing - F1 score')
plt.xlabel('Models')
plt.ylabel('F1 Score')

plt.savefig('./f1.jpg')
plt.show()

