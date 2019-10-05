import matplotlib.pyplot as plt
from matplotlib import patches
import os
import sys

sys.path.append('.')

data_dir = 'data/train'
annotate_file = open('annotate.txt', 'w')
plot = False


def add_to_file(image, xmin, ymin, xmax, ymax):
    if plot:
        fig = plt.figure()
        # add axes to the image
        ax = fig.add_axes([0, 0, 1, 1])
        width = xmax - xmin
        height = ymax - ymin
        # assign different color to different classes of objects
        edgecolor = 'r'
        ax.annotate('', xy=(xmax - 40, ymin + 20))
        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
        plt.imshow(image)
        plt.show()

    ann_object = f'{os.path.abspath(image_file)},{xmin},{ymin},{xmax},{ymax},button\n'
    annotate_file.write(ann_object)


def load_im_data():
    # for r, d, files in os.walk(data_dir):
    for i, f in enumerate(os.listdir(data_dir)):
        if '.xml' in f:
            file_path = os.path.join(data_dir, f)
            xml_file = open(file_path)

            image_file = file_path[:-3] + 'png'
            image = plt.imread(image_file)

            done_rectangle = False
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            for row in xml_file:
                if 'xmin' in row:
                    xmin = int(row.split('>')[1].split('<')[0])
                elif 'ymin' in row:
                    ymin = int(row.split('>')[1].split('<')[0])
                elif 'xmax' in row:
                    xmax = int(row.split('>')[1].split('<')[0])
                elif 'ymax' in row:
                    ymax = int(row.split('>')[1].split('<')[0])
                    done_rectangle = True

                if done_rectangle:
                    add_to_file(image, xmin, ymin, xmax, ymax)
                    done_rectangle = False
def label_text_buttons(img_data):
    filepath = img_data['filepath']
    img_name = filepath.split('/')[-1]
    res = process_text_analysis(b_name, img_name)
    # img1 = cv2.imread(filepath)

    for b in img_data['bboxes']:
        rx1, ry1, rx2, ry2 = b['x1'], b['y1'], b['x2'], b['y2']
        # cv2.rectangle(img1, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        bt_word = ''
        for word in res:
            dx1, dy1, dx2, dy2 = word[1]
            if dx1 > rx1 and dy1 > ry1 and dx2 < rx2 and dy1 < ry2:
                # cv2.putText(img1, word[0], (rx1, ry1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                bt_word += f'{word[0]} '
            elif word[0] != ' ' and word[0] not in button_txt and word[0] not in backgraund_txt:
                backgraund_txt.append(word[0])
                # print(f'{word[0]}, bg')

        bt_word = bt_word[:-1]
        if bt_word != ' ' and bt_word not in button_txt:
            button_txt.append(bt_word)
            # print(f'{bt_word}, bt')

    return backgraund_txt, button_txt


def label_text_buttons(img_data):
    filepath = img_data['filepath']
    img_name = filepath.split('/')[-1]
    res = process_text_analysis(b_name, img_name)
    # img1 = cv2.imread(filepath)

    for b in img_data['bboxes']:
        rx1, ry1, rx2, ry2 = b['x1'], b['y1'], b['x2'], b['y2']
        # cv2.rectangle(img1, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        bt_word = ''
        for word in res:
            dx1, dy1, dx2, dy2 = word[1]
            if dx1 > rx1 and dy1 > ry1 and dx2 < rx2 and dy1 < ry2:
                # cv2.putText(img1, word[0], (rx1, ry1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                bt_word += f'{word[0]} '
            elif word[0] != ' ' and word[0] not in button_txt and word[0] not in backgraund_txt:
                backgraund_txt.append(word[0])
                # print(f'{word[0]}, bg')

        bt_word = bt_word[:-1]
        if bt_word != ' ' and bt_word not in button_txt:
            button_txt.append(bt_word)
            # print(f'{bt_word}, bt')

    X_text = []
    y_text = []

    for bt in button_txt:
        if bt in backgraund_txt:
            backgraund_txt.remove(bt)
        X_text.append(bt)
        y_text.append(1)

    for bg in backgraund_txt:
        if bg:
            X_text.append(bg)
            y_text.append(0)

    return backgraund_txt, button_txt

def load_text_data():
    button_txt = []
    backgraund_txt = []

    for _ in range(len(train_imgs)):
        X, Y, img_data = next(data_gen_train)
        label_text_buttons(img_data)

    for _ in range(len(val_imgs)):
        X, Y, img_data = next(data_gen_val)
        label_text_buttons(img_data)

    text_data = open('/homes/zahara/PycharmProjects/text_button_detection/data/text_bt.txt', 'w')

    for bt in button_txt:
        if bt in backgraund_txt:
            backgraund_txt.remove(bt)
        # print(f'{bt}, bt')
        text_data.write(f'{bt}, bt\n')

    for bg in backgraund_txt:
        if bg:
            text_data.write(f'{bg}, bg\n')
            # print(f'{bg}, bg')

    text_data.close()
    print('Done text annotation')


def text_classification():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Input
    from keras.models import Model

    from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
    np.random.seed(7)

    df = pd.read_csv('/homes/zahara/PycharmProjects/text_button_detection/data/text_bt.txt', delimiter=',')
    df = df.sample(frac=1).reset_index(drop=True)

    X = list(df['text'])
    y = list(df['tag'])
    y = [0 if t == 'bg' else 1 for t in y]

    sentences_train, sentences_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tokenizer = Tokenizer(num_words=1100)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    maxlen = 5

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


    embedding_dim = 50

    inputs = Input(shape=(5,))
    x = Embedding(vocab_size, embedding_dim, input_length=maxlen)(inputs)
    x = Conv1D(128, 5, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train,
                    epochs=5,
                    verbose=False,
                    batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    return


if __name__ == '__main__':
    # load_im_data()
    load_text_data()
