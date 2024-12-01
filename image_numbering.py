import os

image_path = "datasets/test/images"
image_names = os.listdir(image_path)

label_path = "datasets/test/labels"
label_names = os.listdir(label_path)

num = 120
for image_name in image_names:
    for label_name in label_names:
        if label_name[:-4] == image_name[:-4]:
            src_image = os.path.join(image_path, image_name)
            src_label = os.path.join(label_path, label_name)
            dst_image = os.path.join(image_path, '{0:04d}.jpg'.format(num))
            dst_label = os.path.join(label_path, '{0:04d}.txt'.format(num))
            os.rename(src_image, dst_image)
            os.rename(src_label, dst_label)
    num += 1

