import re
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


def read_file(file_name):
    data_list = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            label = " ".join(line[1 : line.find("]")].strip().split())
            text = line[line.find("]") + 1 :].strip()
            data_list.append([label, text])
    return data_list


def ngram(token, n):
    output = []
    for i in range(n - 1, len(token)):
        ngram = " ".join(token[i - n + 1 : i + 1])
        output.append(ngram)
    return output


def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()

    text_alphanum = re.sub("[^a-z0-9#]", " ", text)
    for n in range(nrange[0], nrange[1] + 1):
        text_features += ngram(text_alphanum.split(), n)

    text_punc = re.sub("[a-z0-9]", " ", text)
    text_features += ngram(text_punc.split(), 1)

    return Counter(text_features)


def convert_label(item, name):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += name[idx] + " "

    return label.strip()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--text",
        type=str,
        default="My car skidded on the wet street",
        help="Input text to emojize",
    )
    argparser.add_argument(
        "--filename", type=str, default="./psychExp.txt", help="Input text to emojize"
    )
    # argparser.add_argument('--num_emoji', type=int, default=2, help="Number of emojis to print")
    args = argparser.parse_args()

    # text = "Thank you for dinner!"
    # text = "I don't like it"
    # text = "My car skidded on the wet street"
    # text = "My cat died"

    psychExp_txt = read_file(args.filename)
    emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

    X_all = []
    y_all = []
    for label, text in psychExp_txt:
        y_all.append(convert_label(label, emotions))
        X_all.append(create_feature(text, nrange=(1, 4)))

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=123
    )

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    lsvc = LinearSVC(max_iter=1000, C=1, tol=0.1)
    lsvc.fit(X_train, y_train)

    print(f"Validation acc: {lsvc.score(X_train, y_train)}")
    print("Training acc: {}".format(accuracy_score(y_train, lsvc.predict(X_train))))
    print("Test acc    : {}".format(accuracy_score(y_test, lsvc.predict(X_test))))

    emoji_dict = {
        "joy": "😂",
        "fear": "😱",
        "anger": "😠",
        "sadness": "😢",
        "disgust": "😒",
        "shame": "😳",
        "guilt": "😳",
    }

    features = create_feature(args.text)
    features = vectorizer.transform(features)
    prediction = lsvc.predict(features)[0]
    print(f"{emoji_dict[prediction]} {args.text}")
