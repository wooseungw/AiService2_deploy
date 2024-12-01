
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv



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

