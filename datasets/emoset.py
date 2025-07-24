import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
# Pillow
from PIL import Image

# data_root = 'EmoSet-118K'

class EmoSet(Dataset):
    def __init__(self,
                 data_root,
                 num_emotion_classes,
                 phase,
                 ):
        # Parameter 제한
        assert num_emotion_classes in (8, 2)
        assert phase in ('train', 'val', 'test')

        self.transforms_dict = self.get_data_transforms()

        # info.json
        self.info = self.get_info(data_root, num_emotion_classes)
        
        # data.json
        if phase == 'train':
            self.transform = self.transforms_dict['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
        else:
            raise NotImplementedError

        data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
        self.data_store = [
            [
                self.info['emotion']['label2idx'][item[0]], # 정수 label
                item[1], # 이미지 경로
                os.path.join(data_root, item[2]), # Annotation 경로 (사용 X)
            ]
            for item in data_store
        ]

    @classmethod
    # 이미지 전처리
    def get_data_transforms(cls):
        # .Compose에 정의된 순서대로 실행
        transforms_dict = {
            'train': transforms.Compose([
                # 랜덤하게 Crop후 사이즈 맞춤
                transforms.RandomResizedCrop(224),
                # 50프로 확률로 수평 방향으로 Image flip
                transforms.RandomHorizontalFlip(),
                # Image to Tensor (HWC -> CHW)
                transforms.ToTensor(),
                # Image Normalization (mean, std)
                # ImageNet 기반 Nomralization (R, G, B 순서로)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # 짧은 변을 224 size로 맞추고 나머지 긴 변은 기존 비율에 맞게 변경
                transforms.Resize(224),
                # 가운데 기준 (224, 244) 영역을 잘라냄
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalization은 Tensor를 입력받기 때문에
                # ToTensor() 뒤에 위치해야 함
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return transforms_dict

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (8, 2)

        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 8:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 1,
                    'contentment': 2,
                    'excitement': 3,
                    'anger': 4,
                    'disgust': 5,
                    'fear': 6,
                    'sadness': 7,
                },
                'idx2label': {
                    "0": "amusement",
                    "1": "awe",
                    "2": "contentment",
                    "3": "excitement",
                    "4": "anger",
                    "5": "disgust",
                    "6": "fear",
                    "7": "sadness"
                }
            }
            info['emotion'] = emotion_info
        # 아마 사용하지 않을 것 같지만 혹시 몰라서 놔둠
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'disgust': 1,
                    'fear': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image) # Image 전처리
        return image

    def __getitem__(self, item):
        emotion_label_idx, image_path, _ = self.data_store[item]
        # 추가 경로 설정
        image_path = os.path.join('data', 'EmoSet-118K', image_path)
        os.path.abspath(image_path)

        image = self.load_image_by_path(image_path)
        data = {'image': image, 'emotion_label_idx': emotion_label_idx}
        # keys: image, emotion_label_idx
        # values: 전체 이미지의 정보, 각 이미지의 label
        return data

    def __len__(self):
        return len(self.data_store)
    
    

        
