import pandas as pd

useCols = ["id", "name", "type", "flavour", "process", "price", "gluten_free", "category"]

food_path = "https://docs.google.com/spreadsheets/d/1x1IJDBVuGd1EXlHdomFJXcYvCJpecF561YxinLoweA4/export?format=csv&gid=0"

def load_data():
    try:
        food_data = pd.read_csv(food_path)
        print('Loaded data')
        return food_data
    except FileNotFoundError as e:
        print('File not found')
        return None


def preprocess_data(food_data):
    print('Preprocessing data')
    if food_data is None:
        print('No data')
        return None
    else:
        data_columns = food_data.columns
        data_rows = food_data.values
        data = pd.DataFrame(food_data)
        print('Preprocessing data finished')
        return data


def init():
    raw_data = load_data()
    data = preprocess_data(raw_data)
    return data

