from sklearn.model_selection import train_test_split   # hàm chia train và test
from sklearn.preprocessing import LabelBinarizer
import pickle
# Mở tệp để đọc dữ liệu
file = open('data_file.csv', 'rb')
data = pickle.load(file)
file.close()
x_data , y_data = data

encoder = LabelBinarizer()
y_data=encoder.fit_transform(y_data)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=81)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout


# Xây dựng mô hình LSTM
model = Sequential()
model.add(Conv1D(2048, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(52, 1)))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(52, 1)))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128))


model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# In thông tin về kiến trúc mô hình
model.summary()
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))