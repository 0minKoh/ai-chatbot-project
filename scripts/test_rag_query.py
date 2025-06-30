# ai_chatbot_project/scripts/test_rag_query.py
import os
import sys # sys 모듈 임포트
import django
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Django 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

from chatbot_app.models import FAQ, FAQCategory

# 임베딩 모델 로드 (build_faq_vector_db.py와 동일한 모델 사용)
print("임베딩 모델 로딩 중...")
# build_faq_vector_db.py에서 사용한 모델과 동일해야 합니다.
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
print("임베딩 모델 로드 완료.")

# 벡터 DB가 저장된 디렉토리
VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../vector_dbs')

def load_vector_db(category_name):
    """특정 카테고리에 대한 FAISS 인덱스와 ID 매핑 정보를 로드합니다."""
    index_file_path = os.path.join(VECTOR_DB_DIR, f'faiss_index_{category_name}.bin')
    id_mapping_file_path = os.path.join(VECTOR_DB_DIR, f'id_mapping_{category_name}.pkl')

    if not os.path.exists(index_file_path) or not os.path.exists(id_mapping_file_path):
        print(f"오류: '{category_name}'에 대한 벡터 DB 파일이 존재하지 않습니다.")
        print(f"'{category_name}'에 대한 벡터 DB를 먼저 구축해주세요: python scripts/build_faq_vector_db.py")
        return None, None

    print(f"'{category_name}' 벡터 DB 로드 중...")
    index = faiss.read_index(index_file_path)
    with open(id_mapping_file_path, 'rb') as f:
        faq_ids = pickle.load(f)
    print(f"'{category_name}' 벡터 DB 로드 완료. 총 {index.ntotal}개 벡터.")
    return index, faq_ids

def search_faq(query_text, category_name, top_k=3):
    """주어진 쿼리에 대해 특정 카테고리의 FAQ를 검색합니다."""
    index, faq_ids = load_vector_db(category_name)

    if index is None or faq_ids is None:
        return [] # 벡터 DB를 로드하지 못했으면 빈 리스트 반환

    # 쿼리 임베딩
    query_embedding = model.encode([query_text], convert_to_numpy=True)

    # FAISS 검색
    # D: 거리 (distance), I: 인덱스 (index of the found vectors in the original corpus)
    # L2 distance (유클리드 거리)는 값이 작을수록 유사도가 높습니다.
    D, I = index.search(query_embedding, top_k)

    found_faqs = []
    # I[0]는 첫 번째 쿼리(우리는 쿼리를 하나만 넣었음)에 대한 검색 결과 인덱스입니다.
    for i, faq_idx_in_corpus in enumerate(I[0]):
        if faq_idx_in_corpus == -1: # FAISS가 결과를 찾지 못한 경우 -1을 반환할 수 있음
            continue
        
        # faq_ids 리스트는 FAISS 인덱스와 Django FAQ ID를 매핑합니다.
        faq_id = faq_ids[faq_idx_in_corpus]
        try:
            faq_obj = FAQ.objects.get(id=faq_id)
            found_faqs.append({
                'faq_object': faq_obj,
                'similarity_score': D[0][i] # 검색된 FAQ의 유사도 점수 (거리)
            })
        except FAQ.DoesNotExist:
            print(f"경고: ID {faq_id}에 해당하는 FAQ 객체를 데이터베이스에서 찾을 수 없습니다. (FAQ가 삭제되었을 수 있습니다.)")

    # 유사도 점수가 낮은(거리가 가까운) 순서대로 정렬 (유클리드 거리이므로)
    found_faqs.sort(key=lambda x: x['similarity_score'])
    
    return found_faqs

if __name__ == "__main__":
    print("FAQ 벡터 DB 검색 테스트를 시작합니다.")

    # 테스트할 카테고리 이름을 입력합니다. (Django Admin에 추가한 카테고리 이름과 동일해야 합니다)
    # 예시: '수파자 과외', '수파자 낭독', '소방 히어로 멤버십', '기타 문의'
    # build_faq_vector_db.py에서 CSV로 임포트한 카테고리 중 하나를 사용해 보세요.
    test_category = input("테스트할 카테고리 이름을 입력하세요 (예: 수파자 과외, 기타 문의): ").strip()

    if not test_category:
        print("카테고리 이름이 입력되지 않았습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 입력된 카테고리가 데이터베이스에 존재하는지 확인
    if not FAQCategory.objects.filter(name=test_category).exists():
        print(f"\n오류: '{test_category}' 카테고리가 데이터베이스에 존재하지 않습니다.")
        print("Django Admin에서 해당 카테고리를 먼저 추가하거나, build_faq_vector_db.py 스크립트 실행을 확인해주세요.")
    else:
        while True:
            user_query = input(f"\n'{test_category}' 카테고리에 대한 질문을 입력하세요 (종료하려면 'q' 입력): ").strip()
            if user_query.lower() == 'q':
                break
            if not user_query:
                print("질문을 입력해주세요.")
                continue

            # top_k 값으로 검색 결과 개수 조절
            results = search_faq(user_query, test_category, top_k=2)

            if results:
                print(f"\n검색 결과 (카테고리: {test_category}):")
                for r in results:
                    # 유클리드 거리는 작을수록 유사함
                    print(f"  - 유사도 (거리): {r['similarity_score']:.4f}")
                    print(f"    원본 질문: {r['faq_object'].question}")
                    print(f"    원본 답변: {r['faq_object'].answer}")
            else:
                print(f"'{user_query}'에 대한 관련 FAQ를 '{test_category}' 카테고리에서 찾을 수 없습니다.")

    print("\n검색 테스트 종료.")