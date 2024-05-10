from torch.utils.data import Dataset
from torchvision.io import read_image

def get_images(data_path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(data_path, '*.{}'.format(ext))))
    return files

def load_annoataion(path):
    '''
    load annotation from the text file
    :param path:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(path):
        return np.array(text_polys, dtype=np.float32)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def img_to_score_label(image):
    return image

def img_to_geo_label(image):
    return image

class EastDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = get_images(data_dir)

        self.polys = []
        self.text = []
        for img_path in self.image_paths:
            annotation_path = img_path.split(".")[0] + ".txt"
            text_polys, text_tags = load_annoataion(annotation_path)
            self.polys.append(text_polys)
            self.text.append(text_tags)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __get_item__(self, idx):
        image = read_image(self.image_paths[idx])
        text_polys = self.polys[idx]
        text_tags = self.text[idx]

        score_label = img_to_score_label(image)
        geo_label = img_to_geo_label(image)

        return image, text_polys, text_tags