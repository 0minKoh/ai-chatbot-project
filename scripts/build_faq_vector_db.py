# ai_chatbot_project/scripts/build_faq_vector_db.py
import csv
import os
import sys
import django
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Django 환경 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

from chatbot_app.models import FAQ, FAQCategory

# 임베딩 모델 로드
print("임베딩 모델 로딩 중...")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
print("임베딩 모델 로드 완료.")

# 데이터를 저장할 디렉토리 설정
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# 벡터 DB를 저장할 디렉토리
VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../vector_dbs')
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def _build_vector_db():
    categories = FAQCategory.objects.all()
    print(f"총 {categories.count()}개의 FAQ 카테고리 발견.")

    for category in categories:
        print(f"\n카테고리 '{category.name}'에 대한 벡터 DB 구축 중...")
        faqs = FAQ.objects.filter(category=category)

        if not faqs.exists():
            print(f"  - '{category.name}' 카테고리에 FAQ 데이터가 없습니다. 건너뜜.")
            continue

        # 질문과 답변을 결합하여 임베딩할 텍스트 리스트 생성
        corpus = []
        faq_ids = [] # FAISS 인덱스와 FAQ 객체를 매핑하기 위한 ID 리스트
        for faq in faqs:
            # 질문과 답변을 함께 임베딩하여 문맥 정보를 모두 포함하도록 합니다.
            text_to_embed = f"질문: {faq.question}\n답변: {faq.answer}"
            corpus.append(text_to_embed)
            faq_ids.append(faq.id)

        if not corpus:
            print(f"  - '{category.name}' 카테고리에 유효한 텍스트 데이터가 없습니다. 건너뜜.")
            continue

        print(f"  - 총 {len(corpus)}개의 FAQ를 임베딩 중...")
        embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        print(f"  - 임베딩 완료. 벡터 차원: {embeddings.shape[1]}")

        # FAISS 인덱스 생성
        # IndexFlatL2는 가장 간단한 유클리드 거리 기반의 인덱스입니다.
        # 대규모 데이터셋에서는 IndexHNSWFlat, IndexIVFFlat 등을 고려할 수 있습니다.
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings) # 임베딩을 인덱스에 추가

        print(f"  - FAISS 인덱스에 {index.ntotal}개의 벡터 추가 완료.")

        # FAISS 인덱스 및 매핑 정보 저장
        index_file_path = os.path.join(VECTOR_DB_DIR, f'faiss_index_{category.name}.bin')
        id_mapping_file_path = os.path.join(VECTOR_DB_DIR, f'id_mapping_{category.name}.pkl')

        faiss.write_index(index, index_file_path)
        with open(id_mapping_file_path, 'wb') as f:
            pickle.dump(faq_ids, f)

        print(f"  - 벡터 DB 및 ID 매핑 정보 저장 완료: {index_file_path}, {id_mapping_file_path}")

    print("\n모든 카테고리에 대한 벡터 DB 구축 완료.")

def _import_faq_csv(csv_file_path: str):
    """
    CSV 파일로부터 FAQ 데이터를 임포트합니다.
    CSV 파일 형식: category_name,question,answer
    """
    if not os.path.exists(csv_file_path):
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다.")
        return

    print(f"'{csv_file_path}' 파일에서 FAQ 데이터를 임포트 중...")
    
    imported_count = 0
    skipped_count = 0

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # CSV 헤더가 'category_name', 'question', 'answer'인지 확인
        if not all(col in reader.fieldnames for col in ['category_name', 'question', 'answer']):
            print("오류: CSV 파일 헤더가 'category_name,question,answer' 형식이 아닙니다.")
            return

        for row in reader:
            category_name = row['category_name'].strip()
            question = row['question'].strip()
            answer = row['answer'].strip()

            if not category_name or not question or not answer:
                print(f"경고: 유효하지 않은 데이터가 포함된 행을 건너뜜: {row}")
                skipped_count += 1
                continue

            try:
                # 카테고리 가져오기 또는 생성
                category, created = FAQCategory.objects.get_or_create(name=category_name)
                if created:
                    print(f"  - 새로운 카테고리 생성: '{category_name}'")

                # FAQ 객체 생성 또는 업데이트
                # 중복 방지를 위해 question으로 검색 후 없으면 생성
                faq, created_faq = FAQ.objects.get_or_create(
                    category=category,
                    question=question,
                    defaults={'answer': answer}
                )
                if created_faq:
                    print(f"  - FAQ 추가: '{question[:30]}...' (카테고리: {category_name})")
                    imported_count += 1
                else:
                    # 이미 존재하는 FAQ의 경우, 답변이 다르면 업데이트 (선택 사항)
                    if faq.answer != answer:
                        faq.answer = answer
                        faq.save()
                        print(f"  - FAQ 업데이트: '{question[:30]}...' (카테고리: {category_name})")
                        imported_count += 1 # 업데이트도 임포트로 간주
                    else:
                        print(f"  - FAQ 건너뜀 (이미 존재하고 동일): '{question[:30]}...'")
                        skipped_count += 1

            except Exception as e:
                print(f"오류: FAQ 임포트 중 문제가 발생했습니다: {row} - {e}")
                skipped_count += 1
                
        print(f"\nCSV 임포트 완료. 총 {imported_count}개 FAQ 추가/업데이트, {skipped_count}개 건너뜀.")

# FAQ 데이터를 사용하여 벡터 DB 구축
if __name__ == "__main__":
    print("FAQ 데이터를 사용하여 벡터 DB 구축을 시작합니다.")
    _import_faq_csv(os.path.join(DATA_DIR, 'RAG_supaja_faq.csv'))
    _build_vector_db()
    print("벡터 DB 구축이 완료되었습니다.")
