import pandas as pd

def train_csv_to_dict(train_csv_dir, ext='.jpg'):
    df = pd.read_csv(train_csv_dir)
    filenames = [i + ext for i in df['image_name']]
    tag_list = df['tags'].str.split(' ')
    return dict(zip(filenames, tag_list))
