import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def createDataSet():
    data = []
    labels = []
    with open("data/3.DecisionTree/data.txt") as ifile:
        for line in ifile:
            #特征：身高 体重 label：胖瘦
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])

    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)

    y[labels == 'fat'] = 1
    return x, y

def predict_train(x_train, y_train):
    clf =tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)

    print('feature_importances_:%s' % clf.feature_importances_)

    y_pre = clf.predict(x_train)
    print(y_pre)
    print(y_train)
    print(np.mean(y_pre == y_train))
    return y_pre, clf

def show_precision_recall(x, y, clf, y_train, y_pre):
    precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
    answer = clf.predict_proba(x)[:, 1]

    target_names = ['thin', 'fat']
    print(classification_report(y, answer, target_names=target_names))
    print(answer)
    print(y)


def show_pdf(clf):
    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("../../../output/3.DecisionTree/tree.pdf")

if __name__ == "__main__":
    x, y = createDataSet()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    y_pre, clf = predict_train(x_train, y_train)

    show_precision_recall(x, y, clf, y_train, y_pre)

    show_pdf(clf)
    
