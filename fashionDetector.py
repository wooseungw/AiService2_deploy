import torch
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import json
from yolo_w_multi_classhead import YOLO_MultiClass

# 카테고리별 인코딩
categories = {
    "top": 0, "blouse": 1, "casual_top": 2, "knitwear": 3, "shirt": 4, "vest": 5,
    "coat": 6, "jacket": 7, "jumper": 8, "padding": 9, "jeans": 10, "pants": 11,
    "skirt": 12, "leggings": 13, "dress": 14, "jumpsuit": 15, "swimwear": 16
}

category_encodings = {
    "length": {"short": 0, "midi": 1, "long": 2},
    "color": {
        "black": 0, "white": 1, "gray": 2, "red": 3, "pink": 4, "orange": 5, "beige": 6,
        "brown": 7, "yellow": 8, "green": 9, "khaki": 10, "mint": 11, "blue": 12,
        "navy": 13, "skyblue": 14, "purple": 15, "lavender": 16, "wine": 17,
        "neon": 18, "gold": 19
    },
    "material": {
        "fur": 0, "knit": 1, "mustang": 2, "lace": 3, "suede": 4, "linen": 5,
        "angora": 6, "mesh": 7, "corduroy": 8, "fleece": 9, "sequin": 10, "neoprene": 11,
        "denim": 12, "silk": 13, "jersey": 14, "spandex": 15, "tweed": 16,
        "jacquard": 17, "velvet": 18, "leather": 19, "vinyl": 20, "cotton": 21,
        "wool": 22, "chiffon": 23, "synthetic": 24
    },
    "sleeve_length": {"sleeveless": 0, "3/4_sleeve": 1, "short_sleeve": 2, "long_sleeve": 3, "cap": 4},
    "neckline": {
        "round": 0, "square": 1, "u_neck": 2, "collarless": 3, "v_neck": 4,
        "hood": 5, "halter": 6, "turtleneck": 7, "offshoulder": 8,
        "boatneck": 9, "one_shoulder": 10, "sweetheart": 11
    },
    "fit": {"normal": 0, "skinny": 1, "loose": 2, "wide": 3, "oversized": 4, "tight": 5}
}


class FashionDetector:
    def __init__(self, detection_model_path, attr_model_path, attr_weights_path, categories, category_encodings):
        print(f"Detection model path: {detection_model_path}")
        print(f"Attribute model path: {attr_model_path}")
        print(f"Weight path: {attr_weights_path}")

        # 모델 로드
        self.detection_model = YOLO(detection_model_path)
        self.attr_model = YOLO_MultiClass(attr_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 속성 분류 모델 가중치 로드
        self.attr_model.load_state_dict(torch.load(attr_weights_path, map_location=self.device), strict=False)
        self.attr_model.to(self.device)
        self.attr_model.eval()

        # 카테고리와 속성 인코딩 저장
        self.categories = categories
        self.category_encodings = category_encodings

        # 상위 카테고리
        self.top_level_categories = {
            "outer": ["coat", "jacket", "jumper", "padding"],
            "top": ["top", "blouse", "casual_top", "knitwear", "shirt", "vest"],
            "bottom": ["jeans", "pants", "skirt", "leggings"],
            "onepiece": ["dress", "jumpsuit", "swimwear"]
        }

        # 클래스 인덱스 -> 카테고리 이름
        self.reversed_categories = {v: k for k, v in categories.items()}

    def preprocess_image(self, image_path):
        """
        속성 분류 모델용 이미지 전처리.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def get_attribute_predictions(self, image):
        """
        속성 분류 모델을 이용한 속성 예측.
        """
        with torch.no_grad():
            return self.attr_model(image)

    def classify_top_level(self, category_name):
        """
        Classify the top-level category (outer, top, bottom, onepiece).
        """
        for key, subcategories in self.top_level_categories.items():
            if category_name in subcategories:
                print(f"Category '{category_name}' classified as top-level '{key}'")
                return key
        print(f"Category '{category_name}' could not be classified")
        return None


    def process(self, image_path):
        """
        이미지 처리 후 카테고리와 속성 결과 반환.
        """
        # Detection 모델 예측
        detection_result = self.detection_model(image_path)
        detections = detection_result[0].boxes

        # 디버깅: Detection 결과 확인
        print(f"Detected {len(detections.cls)} objects")
        print(f"Detection Classes: {[int(cls.item()) for cls in detections.cls]}")

        # 속성 분류 모델 예측
        preprocessed_image = self.preprocess_image(image_path)
        attribute_predictions = self.get_attribute_predictions(preprocessed_image)

        # 디버깅: 속성 분류 결과 확인
        print(f"Attribute Predictions: {attribute_predictions}")

        # 결과 저장
        results = {"outer": [], "top": [], "bottom": [], "onepiece": []}

        for i in range(len(detections.cls)):
            cls_idx = int(detections.cls[i].item())
            conf = detections.conf[i].item()
            bbox = detections.xyxy[i].tolist()

            # 클래스 인덱스를 통해 카테고리 이름 조회
            category_name = self.reversed_categories.get(cls_idx, "Unknown")
            print(f"Class Index: {cls_idx}, Category Name: {category_name}")

            # 상위 카테고리 결정
            top_level_category = self.classify_top_level(category_name)
            if not top_level_category:
                print(f"Warning: Top-level category not found for {category_name}")
                continue

            # 속성 결과 처리
            obj_attributes = {}
            try:
                # 탐지된 객체의 상위 카테고리에 해당하는 속성 예측값 가져오기
                category_attributes = attribute_predictions.get(top_level_category, {})
                # 각 속성 카테고리에 대해 처리
                for attr_name, encoding in self.category_encodings.items():
                    if attr_name in category_attributes:
                        # 해당 속성의 예측값 가져오기
                        probabilities = category_attributes[attr_name][0]  # 배치 크기가 1이라고 가정
                        # 확률이 가장 높은 인덱스 찾기
                        max_index = torch.argmax(probabilities).item()
                        confidence = probabilities[max_index].item()
                        # 속성 값 매핑
                        attr_value = [k for k, v in encoding.items() if v == max_index]
                        attr_value = attr_value[0] if attr_value else "Unknown"
                        # 속성 추가
                        obj_attributes[attr_name] = {"value": attr_value, "confidence": confidence}
                    else:
                        print(f"속성 '{attr_name}'가 '{top_level_category}'에 없습니다.")
            except Exception as e:
                print(f"클래스 {category_name}의 속성 처리 중 오류 발생: {e}")


            # 결과 저장
            result = {
                "category": category_name,
                "bounding_box": bbox,
                "confidence": conf,
                "attributes": obj_attributes
            }

            print(f"Result for {top_level_category}: {result}")
            results[top_level_category].append(result)

        # 디버깅: 최종 결과 확인
        print(f"Final Results: {results}")
        return results



    def display_results(self, results):
        """
        JSON 형식으로 결과 출력.
        """
        print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    # 모델 경로
    detection_model_path = "./runs/detect/l_640_dropout025_more_cat_3/weights/best.pt"
    attr_model_path = "yolo11l.pt"
    attr_weights_path = "./fashion_classification_l/best_model.pt"

    # FashionDetector 초기화
    detector = FashionDetector(
        detection_model_path=detection_model_path,
        attr_model_path=attr_model_path,
        attr_weights_path=attr_weights_path,
        categories=categories,
        category_encodings=category_encodings
    )

    # 이미지 처리 및 결과 출력
    image_path = "datasets/images/val/000002.jpg"
    results = detector.process(image_path)
    detector.display_results(results)
