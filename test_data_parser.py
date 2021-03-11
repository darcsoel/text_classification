import sys

from data_parser import DataParser


class TestDataParser(DataParser):
    """Parsing test datasets"""

    path = '20news-bydate-test'
    parsed_data_csv_path = 'test_data.csv'


if __name__ == '__main__':
    parser = TestDataParser()
    parser.parse().save_to_csv()
    sys.exit()
