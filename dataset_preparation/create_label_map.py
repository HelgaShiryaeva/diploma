import pandas as pd


def pbtxt_from_classlist(l, pbtxt_path):
    pbtxt_text = ''

    for i, c in enumerate(l):
        pbtxt_text += 'item {\n    id: ' + str(
            i + 1) + '\n    display_name: "' + c + '"\n}\n\n'

    with open(pbtxt_path, "w+") as pbtxt_file:
        pbtxt_file.write(pbtxt_text)


def pbtxt_from_csv(csv_path, pbtxt_path):
    class_list = list(pd.read_csv(csv_path)['class'].unique())
    class_list.sort()

    pbtxt_from_classlist(class_list, pbtxt_path)


def pbtxt_from_txt(txt_path, pbtxt_path):
    # read txt into a list, splitting by newlines
    data = [
        l.rstrip('\n').strip()
        for l in open(txt_path, 'r', encoding='utf-8-sig')
    ]

    data = [l for l in data if len(l) > 0]

    pbtxt_from_classlist(data, pbtxt_path)


if __name__ == "__main__":
    input_type = 'csv'
    input_file = 'data/train_labels.csv'
    output_file = 'data/label.pbtxt'

    if input_type == 'csv':
        pbtxt_from_csv(input_file, output_file)
    elif input_type == 'txt':
        pbtxt_from_txt(input_file, output_file)
