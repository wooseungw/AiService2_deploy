import os
import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터베이스 URL 설정
DATABASE_URL = os.getenv('DATABASE_URL')

# SQLAlchemy 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class APIKey(Base):
    __tablename__ = 'api_keys'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    api_key = Column(String)
    
    user = relationship("User")

# secret.key 파일에서 키를 로드하는 함수
def load_fernet_key():
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
    return Fernet(key)

# Fernet 키 로드 및 암호화 객체 초기화
try:
    cipher_suite = load_fernet_key()
except FileNotFoundError:
    print("secret.key 파일을 찾을 수 없습니다. 키를 생성하거나 올바른 위치에 파일이 있는지 확인하세요.")
    cipher_suite = None

# 비밀번호 해시화
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

# 비밀번호 검증
def check_password(password, hashed):
    if isinstance(hashed, str):
        hashed = hashed.encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# API 키 암호화
def encrypt_api_key(api_key):
    if cipher_suite is None:
        print("Cipher suite is not initialized. Cannot encrypt the API key.")
        return None
    return cipher_suite.encrypt(api_key.encode('utf-8'))

# API 키 복호화
def decrypt_api_key(encrypted_key):
    if cipher_suite is None:
        print("Cipher suite is not initialized. Cannot decrypt the API key.")
        return None
    try:
        return cipher_suite.decrypt(encrypted_key).decode('utf-8')
    except InvalidToken:
        print("Invalid encryption key.")
        return None

# 데이터베이스 초기화
def init_db():
    Base.metadata.create_all(bind=engine)

# 사용자 추가
def add_user(db, username, password):
    hashed_password = hash_password(password)
    db_user = User(username=username, password=hashed_password)
    try:
        db.add(db_user)
        db.commit()
    except IntegrityError:
        db.rollback()
        print("User already exists.")

# 사용자 검색
def get_user(db, username):
    return db.query(User).filter(User.username == username).first()

# API 키 추가
def add_api_key(db, user_id, api_key):
    encrypted_key = encrypt_api_key(api_key)
    db_api_key = APIKey(user_id=user_id, api_key=encrypted_key)
    db.add(db_api_key)
    db.commit()

# API 키 검색
def get_api_key(db, user_id):
    db_api_key = db.query(APIKey).filter(APIKey.user_id == user_id).first()
    if db_api_key:
        return decrypt_api_key(db_api_key.api_key)
    return None