import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn import tree

golf_df = pd.read_csv('../datasets/1-golf.csv')
one_hot_data = pd.get_dummies(golf_df[['Outlook', 'Temperature', 'Humidity', 'Windy']])
print(golf_df.head())

clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(one_hot_data, golf_df['Play'])

dot_data = tree.export_graphviz(clf_train, out_file=None,
                                feature_names=list(one_hot_data.columns.values),
                                class_names=['Not_Play', 'Play'],
                                rounded=True, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
