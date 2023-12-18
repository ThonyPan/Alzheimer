from utils import *

data = load_dataset('/data/pty/Scripts/course/Big Data Analysis/Project/data/train_pre_data.h5')
# visualize_data(data[0][0], 'visualized_data')

export_ex2_data(data[0][0], 'output.vti')