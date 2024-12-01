import os

data_path = 'datasets/'
changed_labels = [0, -1, 1, -1, 2, 3, -1, 4, 5, 6]
for type in ['test', 'train', 'valid']:
    label_path = data_path + type + '/labels'
    file_list = os.listdir(label_path)
    for file in file_list:
        f = open(os.path.join(label_path, file), 'r')
        lines = f.readlines()
        new_lines = []
        for line in lines:
            changed_label = changed_labels[int(line[0])]
            if changed_label != -1:
                new_lines.append(str(changed_label) + line[1:])
        w = open(os.path.join(label_path, file), 'w')
        for line in new_lines:
            w.write(line)
        w.close()
