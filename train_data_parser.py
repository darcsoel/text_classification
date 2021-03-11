import sys

from data_parser import DataParser


class TrainDataParser(DataParser):
    """Parsing train datasets"""

    path = '20news-bydate-train'
    parsed_data_csv_path = 'train_data.csv'


if __name__ == '__main__':
    parser = TrainDataParser()
    parser.parse().save_to_csv()
    sys.exit()
