import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import wandb
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib
import numpy as np
import seaborn as sns

# 한글 폰트 경로 설정 (Windows의 경우)
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
# 유니코드 마이너스 설정 (마이너스 기호 깨짐 방지)
matplotlib.rcParams['axes.unicode_minus'] = False


# 카테고리별 인코딩 (한국어)
category_encodings = {
    
    "기장": {
        "숏": 0, "미디": 1, "롱": 2
    },
    "색상": {
        "블랙": 0, "화이트": 1, "그레이": 2, "레드": 3, "핑크": 4,
        "오렌지": 5, "베이지": 6, "브라운": 7, "옐로우": 8,
        "그린": 9, "카키": 10, "민트": 11, "블루": 12, "네이비": 13,
        "스카이블루": 14, "퍼플": 15, "라벤더": 16, "와인": 17, 
        "네온": 18, "골드": 19
    },
    "디테일": {
        "비즈": 0, "퍼트리밍": 1, "단추": 2, "글리터": 3, "니트꽈배기": 4,
        "체인": 5, "컷오프": 6, "더블브레스티드": 7, "드롭숄더": 8, 
        "자수": 9, "프릴": 10, "프린지": 11, "플레어": 12, "퀼팅": 13, 
        "리본": 14, "롤업": 15, "러플": 16, "셔링": 17, "슬릿": 18,
        "스팽글": 19, "스티치": 20, "스터드": 21, "폼폼": 22, "포켓": 23,
        "패치워크": 24, "페플럼": 25, "플리츠": 26, "집업": 27, 
        "디스트로이드": 28, "드롭웨이스트": 29, "버클": 30, "컷아웃": 31,
        "X스트랩": 32, "비대칭": 33
    },
    "프린트": {
        "체크": 0, "플로럴": 1, "스트라이프": 2, "레터링": 3, 
        "해골": 4, "타이다이": 5, "지브라": 6, "도트": 7, 
        "카무플라쥬": 8, "그래픽": 9, "페이즐리": 10, "하운즈 투스": 11, 
        "아가일": 12, "깅엄": 13
    },
    "소재": {
        "퍼": 0, "니트": 1, "무스탕": 2, "레이스": 3, "스웨이드": 4,
        "린넨": 5, "앙고라": 6, "메시": 7, "코듀로이": 8, "플리스": 9,
        "시퀸/글리터": 10, "네오프렌": 11, "데님": 12, "실크": 13,
        "저지": 14, "스판덱스": 15, "트위드": 16, "자카드": 17, 
        "벨벳": 18, "가죽": 19, "비닐/PVC": 20, "면": 21,
        "울/캐시미어": 22, "시폰": 23, "합성섬유": 24
    },
    "소매기장": {
        "민소매": 0, "7부소매": 1, "반팔": 2, "긴팔": 3, "캡": 4
    },
    "넥라인": {
        "라운드넥": 0, "스퀘어넥": 1, "유넥": 2, "노카라": 3, 
        "브이넥": 4, "후드": 5, "홀터넥": 6, "터틀넥": 7,
        "오프숄더": 8, "보트넥": 9, "원 숄더": 10, "스위트하트": 11
    },
    "카라": {
        "셔츠칼라": 0, "피터팬칼라": 1, "보우칼라": 2, "너치드칼라": 3,
        "세일러칼라": 4, "차이나칼라": 5, "숄칼라": 6, "테일러드칼라": 7,
        "폴로칼라": 8, "밴드칼라": 9
    },
    "핏": {
        "노멀": 0, "스키니": 1, "루즈": 2, "와이드": 3,
        "오버사이즈": 4, "타이트": 5
    }
}

# 한국어에서 영어로 변환하는 매핑
attribute_translation = {
    
    "기장": "length",
    "색상": "color",
    "디테일": "detail",
    "프린트": "print",
    "소재": "material",
    "소매기장": "sleeve_length",
    "넥라인": "neckline",
    "카라": "collar",
    "핏": "fit"
}

# 카테고리 이름을 영어로 변환하는 매핑
category_translation = {
    "아우터": "outer",
    "상의": "top",
    "하의": "bottom",
    "원피스": "onepiece"
}

def process_labels_to_list(label_data, encodings, attr_translation, cat_translation):
    # 카테고리별 유효한 속성 정의
    valid_attributes = {
        "아우터": ["기장", "색상", "소매기장", "소재", "프린트", "넥라인", "핏"],
        "상의": ["기장", "색상", "소매기장", "소재", "프린트", "넥라인", "핏"],
        "하의": ["기장", "색상", "소재", "프린트", "핏"],
        "원피스": ["기장", "색상", "소매기장", "소재", "프린트", "넥라인", "핏"]
    }

    # 기본 구조 초기화
    processed_labels = {cat_translation[category]: [-1] * len(valid_attributes[category])
                        for category in valid_attributes}

    # 데이터 파싱 및 처리
    for category, category_data in label_data.items():
        if category not in valid_attributes:
            continue  # 유효하지 않은 카테고리 무시

        for idx, attr in enumerate(valid_attributes[category]):
            value = category_data.get(attr, None)

            if value is None:
                encoded_value = -1  # null 값 처리
            elif isinstance(value, list):
                # 리스트의 첫 번째 값 사용 (없으면 -1)
                encoded_value = encodings.get(attr, {}).get(value[0], -1) if value else -1
            else:
                # 단일 값 인코딩
                encoded_value = encodings.get(attr, {}).get(value, -1)

            # 저장
            processed_labels[cat_translation[category]][idx] = encoded_value

    return processed_labels

class FashionDataset(Dataset):
    def __init__(self, img_dir, label_dir, encodings, attr_translation, cat_translation, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.encodings = encodings
        self.attr_translation = attr_translation
        self.cat_translation = cat_translation
        self.transform = transform
        self.img_list = sorted(os.listdir(img_dir))
        self.label_list = sorted(os.listdir(label_dir))

        # 디버깅용 출력
        print(f"Dataset initialized with {len(self.img_list)} images and {len(self.label_list)} labels.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # 이미지 경로 및 검증
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # 라벨 경로 및 검증
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # 이미지 로드 및 전처리
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # PIL 이미지로 변환
        if self.transform:
            image = self.transform(image)

        # 라벨 로드
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error reading label file {label_path}: {e}")

        # 라벨 처리
        processed_labels = process_labels_to_list(
            label_data, self.encodings, self.attr_translation, self.cat_translation
        )
        processed_labels = {key: torch.tensor(value, dtype=torch.long) for key, value in processed_labels.items()}
        return image, processed_labels
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)

class Head(nn.Module):
    def __init__(self, input_channels):
        super(Head, self).__init__()

        # 공통 속성 선언 (가중치 공유)
        self.shared_heads = nn.ModuleDict({
            
            "length": Classify(input_channels, len(category_encodings["기장"])),
            "color": Classify(input_channels, len(category_encodings["색상"])),
            "sleeve_length": Classify(input_channels, len(category_encodings["소매기장"])),
            "material": Classify(input_channels, len(category_encodings["소재"])),
            "print": Classify(input_channels, len(category_encodings["프린트"])),
            "neckline": Classify(input_channels, len(category_encodings["넥라인"])),
            "fit": Classify(input_channels, len(category_encodings["핏"]))
        })

        # 아우터 속성 선언
        self.outer_head = nn.ModuleDict({
            key: self.shared_heads[key] for key in ["length", "color", "sleeve_length", "material", "print", "neckline", "fit"]
        })

        # 상의 속성 선언
        self.top_head = nn.ModuleDict({
            key: self.shared_heads[key] for key in ["length", "color", "sleeve_length", "material", "print", "neckline", "fit"]
        })

        # 하의 속성 선언
        self.bottom_head = nn.ModuleDict({
            key: self.shared_heads[key] for key in ["length", "color", "material", "print", "fit"]
        })

        # 원피스 속성 선언
        self.onepiece_head = nn.ModuleDict({
            key: self.shared_heads[key] for key in ["length", "color", "sleeve_length", "material", "print", "neckline", "fit"]
        })

    def forward(self, x):
        # 각 분류별 출력 (공통 가중치 사용)
        outer_outputs = {key: head(x) for key, head in self.outer_head.items()}
        top_outputs = {key: head(x) for key, head in self.top_head.items()}
        bottom_outputs = {key: head(x) for key, head in self.bottom_head.items()}
        onepiece_outputs = {key: head(x) for key, head in self.onepiece_head.items()}

        return {
            'outer': outer_outputs,
            'top': top_outputs,
            'bottom': bottom_outputs,
            'onepiece': onepiece_outputs
        }

class YOLO_MultiClass(nn.Module):
    def __init__(self, yolo_dir='yolo11s.pt'):
        super().__init__()
        backbone_layers = list(YOLO(yolo_dir).model.model.children())[:11]
        self.backbone = nn.Sequential(*backbone_layers)
        self.head = Head(512)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def compute_loss(outputs, targets):
    """
    손실 함수: 각 카테고리별로 CrossEntropyLoss를 계산하고 평균 손실 반환.
    """
    criterion_ce = nn.CrossEntropyLoss()

    def compute_category_loss(output, target):
        """
        개별 속성에 대한 손실 계산.
        """
        mask = target != -1  # -1 값 제외
        if mask.sum() == 0:  # 유효한 값이 없으면 손실 0 반환
            return torch.tensor(0.0, dtype=torch.float, device=target.device, requires_grad=True)

        valid_targets = target[mask]
        valid_outputs = output[mask]

        # 출력 클래스 범위 검증
        num_classes = valid_outputs.shape[1]
        if valid_targets.max() >= num_classes:
            raise ValueError(f"Target value {valid_targets.max().item()} exceeds number of classes {num_classes}")

        return criterion_ce(valid_outputs, valid_targets.long())

    # 각 카테고리별 손실 계산
    category_losses = {}

    for category in ['outer', 'top', 'bottom', 'onepiece']:
        if category in outputs:
            category_losses[category] = sum(
                compute_category_loss(outputs[category][key], targets[category][:, idx])
                for idx, key in enumerate(outputs[category])
            )
            # 유효한 속성 개수로 나눔
            category_losses[category] /= len(outputs[category])

    # 카테고리별 손실 합산
    total_loss = sum(category_losses.values()) / len(category_losses)

    return total_loss

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pt'):
        """
        Args:
            patience (int): Improvement epochs to wait before stopping.
            delta (float): Minimum change to qualify as improvement.
            path (str): Path for saving the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch_loss, model):
        # Check if the current loss improves the best_loss by at least delta
        if epoch_loss < self.best_loss - self.delta:
            self.best_loss = epoch_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Save the best model checkpoint."""
        
        torch.save(model.state_dict(), self.path)
        print(f"Model saved to {self.path}.")

def plot_confusion_matrix(cm, class_names, title, save_path):
    """혼동 행렬을 시각화하고 저장"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    
def compute_metrics(outputs, targets):
    """각 속성별로 혼동 행렬과 F1 스코어를 계산"""
    metrics = {}
    reverse_attribute_translation = {v: k for k, v in attribute_translation.items()}

    for head in outputs:
        metrics[head] = {}
        for idx, key in enumerate(outputs[head]):
            preds = outputs[head][key].argmax(dim=1)
            trues = targets[head][:, idx]

            # 유효한 샘플만 선택 (-1이 아닌 값)
            valid_indices = trues != -1
            preds = preds[valid_indices]
            trues = trues[valid_indices]

            if len(trues) == 0:
                continue  # 유효한 샘플이 없으면 건너뜁니다.

            # 영어 키를 한국어로 변환
            kor_key = reverse_attribute_translation.get(key, None)
            if kor_key is None:
                print(f"Warning: '{key}'에 대한 번역이 없습니다. 이 속성의 메트릭 계산을 건너뜁니다.")
                continue

            # 클래스 수 가져오기
            num_classes = outputs[head][key].shape[1]

            # 혼동 행렬 계산
            cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
            for t, p in zip(trues, preds):
                cm[t.long(), p.long()] += 1

            # F1 스코어 계산
            tp = torch.diag(cm).float()
            precision = tp / cm.sum(dim=0).float().clamp(min=1)
            recall = tp / cm.sum(dim=1).float().clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
            f1_mean = f1.mean().item()

            metrics[head][key] = {
                'confusion_matrix': cm.numpy(),
                'f1_score': f1_mean,
            }
    return metrics

def validate_model_with_metrics(model, dataloader, compute_loss):
    """
    검증 데이터셋을 사용하여 모델의 성능을 평가하는 함수.
    """
    model.eval()  # Evaluation 모드
    device = next(model.parameters()).device
    running_loss = 0.0
    all_outputs = {'outer': [], 'top': [], 'bottom': [], 'onepiece': []}
    all_targets = {'outer': [], 'top': [], 'bottom': [], 'onepiece': []}

    with torch.no_grad():  # Gradient 계산 비활성화
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}  # 라벨 GPU로 전송

            outputs = model(images)
            loss = compute_loss(outputs, labels)  # Validation Loss 계산
            running_loss += loss.item()

            for category in all_outputs.keys():
                all_outputs[category].append(outputs[category])
                all_targets[category].append(labels[category])

    val_loss = running_loss / len(dataloader)

    # 모든 배치의 출력과 타겟을 결합
    combined_outputs = {k: {key: torch.cat([batch[key] for batch in all_outputs[k]], dim=0) for key in all_outputs[k][0]} for k in all_outputs}
    combined_targets = {k: torch.cat(all_targets[k], dim=0) for k in all_targets}

    metrics = compute_metrics(combined_outputs, combined_targets)

    return val_loss, metrics

    # 모델 학습 함수 수정
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5, project_name="fashion_classification"):
    # Initialize wandb
    wandb.init(project=project_name, config={
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "patience": patience
    })
    wandb.watch(model, log="all", log_freq=10)  # Watch model parameters and gradients

    early_stopping = EarlyStopping(patience=patience, path=os.path.join(project_name,'best_model.pt'))
    device = next(model.parameters()).device  # Ensure device consistency

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}  # 라벨 GPU로 전송

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Training Loss 계산
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}")

        # Validation
        val_loss, metrics = validate_model_with_metrics(model, val_loader, criterion)
        reverse_attribute_translation = {v: k for k, v in attribute_translation.items()}
        for head in metrics:
            for attr in metrics[head]:
                metric = metrics[head][attr]
                # 영어 속성을 한국어로 변환
                kor_attr = reverse_attribute_translation.get(attr, attr)
                print(f"Metrics for {head} - {kor_attr}:")
                print(f"F1 Score: {metric['f1_score']}")
                # 클래스 이름 가져오기
                #class_names = list(category_encodings.get(kor_attr, {}).keys())
                #save_path = f"confusion_matrix_{head}_{attr}.png"
                #plot_confusion_matrix(metric['confusion_matrix'], class_names, f"Confusion Matrix for {head} - {kor_attr}", save_path)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metrics": metrics
        })

        # Early Stopping 및 최고 성능 모델 저장
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            wandb.log({"early_stop": epoch + 1})
            break

    wandb.finish()  # End wandb run
    print('Training complete')


# if __name__ == '__main__':
#     # 이미지 전처리 정의
#     transform = transforms.Compose([
#         transforms.Resize((640, 640)),
#         transforms.ToTensor()
#     ])
#     # 데이터셋 인스턴스 생성
#     train_img_path = './datasets/images/train'
#     train_label_path = './datasets/reg_labels/train'
#     train_dataset = FashionDataset(train_img_path, train_label_path, category_encodings, attribute_translation, category_translation, transform=transform)
#     # 데이터로더 생성
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=24,
#         shuffle=True
#     )
    
#     val_img_path = './datasets/images/val'
#     val_label_path = './datasets/reg_labels/val'
#     val_dataset = FashionDataset(val_img_path, val_label_path, category_encodings, attribute_translation, category_translation, transform=transform)
    
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=24,
#     )
#     weight_path = 'runs\detect\l_640_dropout025_more_cat_3\weights/best.pt'
#     weight_path = 'yolo11l.pt'
#     model = YOLO_MultiClass(weight_path)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)


#     # 옵티마이저 정의
#     optimizer = optim.AdamW(model.parameters(), lr=0.001)

#     # 모델 학습
#     train_model(
#         model=model,
#         train_loader=train_dataloader,
#         val_loader=val_dataloader,
#         criterion=compute_loss,
#         optimizer=optimizer,
#         num_epochs=300,
#         patience=10,
#         project_name="fashion_classification_l"
#     )
    