from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory

from openai import OpenAI
import base64

from db import get_all_clothing_info, get_image_attributes
import json
import requests


# 대화 기록 예시
'''
st.session_state['chat_history'] = [
  {
    "role": "system", 
    "content": [{"type": "text", "text": "You are a helpful assistant."}]
  },
  {
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/36/Danbo_Cheese.jpg"}},
      {"type": "text", "text": "What is this?"}
    ]
  }
]
'''
import requests
import base64

class GPT:
    def __init__(self, api_key, weather, user_info, model="gpt-4o-mini"):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        
        # user_info 기본값 설정 (이전과 동일)
        if not user_info or len(user_info) < 8:
            user_name = "사용자"
            birth_date = "알 수 없음"
            gender = "알 수 없음"
            height = "알 수 없음"
            weight = "알 수 없음"
            personal_color = "알 수 없음"
            mbti = "알 수 없음"
        else:
            user_name = user_info[1]
            birth_date = user_info[2]
            gender = user_info[3]
            height = user_info[4]
            weight = user_info[5]
            personal_color = user_info[6]
            mbti = user_info[7]

        # DB에서 모든 의류 정보 가져오기
        user_id = user_info[0]
        clothing_info = get_all_clothing_info(user_id)
        clothing_info_str = self._format_clothing_info(clothing_info)
        print(clothing_info_str)
        # 시스템 프롬프트 개선
        SYS_PROMPT = f"""
[시스템 정보]
사용자 이름: {user_name}
사용자 생년월일: {birth_date}
사용자 성별: {gender}
사용자 키: {height}
사용자 체중: {weight}
사용자 퍼스널컬러: {personal_color}
사용자 MBTI: {mbti}

위치: {weather['location']}
온도: {weather['temperature']}°C
날씨: {weather['condition']}

사용자의 의류 정보:
{clothing_info_str}

당신은 개인 맞춤형 패션 스타일리스트입니다.
다음과 같은 기능을 제공합니다:

1. 사용자의 체형, 퍼스널컬러, 날씨를 고려한 코디네이션 추천
2. 선택된 의류 아이템을 기반으로 다양한 스타일링 방법 제안
3. 현재 날씨와 상황에 적합한 의류 추천
4. 사용자의 기존 의류를 활용한 새로운 코디 조합 제안

답변 시 다음 사항을 고려하세요:
- 편안한 말투로 대화하되 전문적인 조언을 제공합니다.
- 실용적이고 구체적인 스타일링 방법을 제시합니다.
- 사용자의 특성과 날씨 조건을 항상 고려합니다.
- 답변은 간단명료하게 작성합니다.
"""
        self.messages = [{'role': 'system', 'content': SYS_PROMPT}]

    def _format_clothing_info(self, clothing_info):
        """의류 정보를 문자열로 포맷팅하는 함수"""
        if not clothing_info:
            return "저장된 의류 정보가 없습니다."
        
        info_str = ""
        for idx, item in enumerate(clothing_info, 1):
            category, bounding_box, confidence, attributes = item
            attributes_dict = json.loads(attributes)
            
            info_str += f"{idx}. 카테고리: {category} (신뢰도: {confidence:.2f})\n"
            for key, value in attributes_dict.items():
                info_str += f"  - {key}: {value['value']} (신뢰도: {value['confidence']:.2f})\n"
            info_str += "\n"
            
        return info_str
    def get_selected_image_info(self, image_path):
        """선택된 이미지의 의류 속성 정보를 가져오는 함수"""
        # DB에서 이미지 속성 정보 조회
        image_attributes = get_image_attributes(image_path)
        if not image_attributes:
            return ""
        
        # 속성 정보를 문자열로 변환
        info_str = "\n[선택된 의류 정보]\n"
        for attr in image_attributes:
            category, _, confidence, attributes = attr
            attributes_dict = json.loads(attributes)
            
            info_str += f"카테고리: {category} (신뢰도: {confidence:.2f})\n"
            for key, value in attributes_dict.items():
                info_str += f"- {key}: {value['value']} (신뢰도: {value['confidence']:.2f})\n"
            info_str += "\n"
            
        return info_str
    def generate(self, text_prompt=None, img_prompt=None):
        if not text_prompt and not img_prompt:
            return "텍스트 또는 이미지를 입력해주세요."
        
        # 컨텐츠 준비
        contents = []
        
        # 이미지가 선택된 경우 의류 속성 정보 추가
        if img_prompt:
            image_info = self.get_selected_image_info(img_prompt)
            # 이미지 정보를 포함한 프롬프트 생성
            if text_prompt:
                text_prompt = f"{image_info}\n사용자 질문: {text_prompt}"
            else:
                text_prompt = f"{image_info}\n이 의류에 대해 설명해주세요."
            
            # 이미지 인코딩 및 추가
            img = self._encode_image(img_prompt)
            if img:
                contents.append({'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{img}"}})
        
        if text_prompt:
            contents.append({'type': 'text', 'text': text_prompt})
        
        self.messages.append({'role': 'user', 'content': contents})
        
        # API 요청 및 응답 처리 (이전과 동일)
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": 300
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                  headers=self.headers, 
                                  json=payload, 
                                  timeout=10)
            response = response.json()
            
            if 'choices' in response:
                assistant_text = response['choices'][0]['message']['content']
                self.messages.append({'role': 'assistant', 'content': assistant_text})
                return assistant_text
            else:
                error_message = response.get('error', {}).get('message', 'Unknown error occurred')
                print("Error:", error_message)
                return f"Error: {error_message}"
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return "API 요청 중 문제가 발생했습니다."
        
    def _encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None


class Chatbot:
    def __init__(self,
                 api_key,
                 weather,
                 user_info,
                 model = "gpt-4o-mini"
                 ):
        self.model = ChatOpenAI(api_key=api_key, model=model)
        # 프롬프트 템플릿 정의
        SYS_PROMPT = f"""
[시스템 정보]
사용자 이름: {user_info[1]}
사용자 생년월일: {user_info[2]}
사용자 성별: {user_info[3]}
사용자 키: {user_info[4]}
사용자 체중: {user_info[5]}
사용자 퍼스널컬러: {user_info[6]}
사용자 MBTI: {user_info[7]}

위치: {weather['location']}
온도: {weather['temperature']}°C
날씨: {weather['condition']}

사용자의 요구에 따라 패션에 대한 정보를 제공하는 챗봇입니다.
편안한 말투로 사용자의 질문에 답변하세요.
사용자와 현재 날씨 정보를 기반으로 추천해야합니다.
의류 트렌드와 사용자에게 어울릴 스타일을 추천해드립니다.
사용자가 읽어야할 텍스트가 너무 많지 않게 해주세요. 가독성을 고려해주세요.

[대화 내역]
{{history}}

[현재 대화]
사용자: {{input}}
어시스턴트:
"""
        self.prompt = ChatPromptTemplate.from_template(SYS_PROMPT)
        self.history = ChatMessageHistory()
        
    def generate(self, input_message):
        # 사용자의 메시지를 히스토리에 추가
        self.history.add_user_message(input_message)
        
        # 히스토리를 문자열로 변환
        history_str = ""
        for msg in self.history.messages[:-1]:  # 현재 메시지를 제외한 히스토리
            if msg.type == "human":
                history_str += f"사용자: {msg.content}\n"
            elif msg.type == "ai":
                history_str += f"어시스턴트: {msg.content}\n"
        
        # 프롬프트 생성
        prompt_input = self.prompt.format_prompt(history=history_str, input=input_message).to_messages()
        
        # 모델로부터 응답 생성
        response = self.model(prompt_input)
        
        # 어시스턴트의 응답을 히스토리에 추가
        self.history.add_ai_message(response.content)
        
        return response.content
