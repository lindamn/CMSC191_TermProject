import numpy as np
import tensorflow as tf
import pandas as pd
import glob

# importing csv files into pandas dataframe ##########################
path = r'Facebook Comment Volume Dataset\Training' 
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, header=None) 
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

# drops unused columns
frame = frame.drop(columns=53)
frame = frame.drop(columns=[46,47,48,49,50,51,52])
frame = frame.drop(columns=38)
frame = frame.drop(columns=[31,32,33])
frame = frame.drop(columns=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])

# adds a column to label rows if they are engaging or not
frame[53] = np.where(np.logical_or(np.logical_and(frame[34] <= 24, frame[29] >= 100), np.logical_and(frame[34] > 24, frame[30] >= 100)), 1, 0)

frame = frame.rename(columns={0:"page_likes", 1:"page_checkins", 2:"page_talkingabout", 3:"page_category",
    29:"c1", 30:"c2", 
    # 31:"c3", 32:"c4", 33:"c5", 
    34:"base_time",
    35:"post_length", 36:"post_sharecount", 37:"post_promotion", 
    # 38:"h_local",
    39:"posted_sunday", 40:"posted_monday", 41:"posted_tuesday", 42:"posted_wednesday", 43:"posted_thursday", 44:"posted_friday", 45:"posted_saturday", 
    # 46:"bt_sunday", 47:"bt_monday", 48:"bt_tuesday", 49:"bt_wednesday", 50:"bt_thursday", 51:"bt_friday", 52:"bt_saturday", 
    53:"target"
})

train=frame.sample(frac=0.8,random_state=191) #random state is a seed value
train_target = train.pop("target")
test=frame.drop(train.index)
test_target = test.pop("target")

train_dataset = tf.convert_to_tensor(train)
test_dataset = tf.convert_to_tensor(test)

# print(train_dataset)
# print(type(train_dataset))

# start of neural network model ######################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(13, activation=tf.nn.sigmoid, input_shape=(17,)),
    tf.keras.layers.Dense(13, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, train_target, epochs=15)

model.evaluate(test_dataset, test_target)

model.summary()

# print(model.trainable_variables) 
print(model.get_weights())