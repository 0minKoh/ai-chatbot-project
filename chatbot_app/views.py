# ai_chatbot_project/chatbot_app/views.py
import json
import requests
import os
import faiss
import numpy as np
import pickle
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt # CSRF 보호 비활성화 (개발용)
from sentence_transformers import SentenceTransformer

from .models import FAQ, FAQCategory

# --- RAG 관련 전역 변수 및 함수 ---
# 임베딩 모델 로드 (FastAPI 서버와 동일한 모델이어야 함)
# Django 앱이 시작될 때 한 번만 로드되도록 합니다.
print("Django 웹 서버: 임베딩 모델 로딩 중...")
try:
    EMBEDDING_MODEL = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    print("Django 웹 서버: 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"Django 웹 서버: 임베딩 모델 로드 실패: {e}")
    EMBEDDING_MODEL = None # 모델 로드 실패 시 None으로 설정

# 벡터 DB가 저장된 디렉토리 (scripts/build_faq_vector_db.py와 동일한 경로)
VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../vector_dbs')

# 로드된 FAISS 인덱스와 ID 매핑을 저장할 딕셔너리
# 각 카테고리별로 인덱스를 한 번만 로드하도록 캐싱합니다.
LOADED_VECTOR_DBS = {}

def load_vector_db_for_category(category_name):
    """특정 카테고리에 대한 FAISS 인덱스와 ID 매핑 정보를 로드하거나 캐시된 것을 반환합니다."""
    if category_name in LOADED_VECTOR_DBS:
        return LOADED_VECTOR_DBS[category_name]['index'], LOADED_VECTOR_DBS[category_name]['faq_ids']

    index_file_path = os.path.join(VECTOR_DB_DIR, f'faiss_index_{category_name}.bin')
    id_mapping_file_path = os.path.join(VECTOR_DB_DIR, f'id_mapping_{category_name}.pkl')

    if not os.path.exists(index_file_path) or not os.path.exists(id_mapping_file_path):
        print(f"오류: '{category_name}'에 대한 벡터 DB 파일이 존재하지 않습니다.")
        return None, None

    print(f"'{category_name}' 벡터 DB 로드 중...")
    try:
        index = faiss.read_index(index_file_path)
        with open(id_mapping_file_path, 'rb') as f:
            faq_ids = pickle.load(f)

        LOADED_VECTOR_DBS[category_name] = {'index': index, 'faq_ids': faq_ids}
        print(f"'{category_name}' 벡터 DB 로드 완료. 총 {index.ntotal}개 벡터.")
        return index, faq_ids
    except Exception as e:
        print(f"'{category_name}' 벡터 DB 로드 중 오류 발생: {e}")
        return None, None

def search_faq_in_vector_db(query_text, category_name, top_k=3):
    """주어진 쿼리에 대해 특정 카테고리의 FAQ를 검색합니다."""
    if EMBEDDING_MODEL is None:
        print("경고: 임베딩 모델이 로드되지 않아 FAQ 검색을 수행할 수 없습니다.")
        return []

    index, faq_ids = load_vector_db_for_category(category_name)

    if index is None or faq_ids is None:
        return []

    query_embedding = EMBEDDING_MODEL.encode([query_text], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)

    found_faqs = []
    for i, faq_idx_in_corpus in enumerate(I[0]):
        if faq_idx_in_corpus == -1:
            continue

        faq_id = faq_ids[faq_idx_in_corpus]
        try:
            faq_obj = FAQ.objects.get(id=faq_id)
            found_faqs.append({
                'faq_object': faq_obj,
                'similarity_score': D[0][i]
            })
        except FAQ.DoesNotExist:
            print(f"경고: ID {faq_id}에 해당하는 FAQ 객체를 DB에서 찾을 수 없습니다.")

    found_faqs.sort(key=lambda x: x['similarity_score']) # 거리가 짧은 순서 (유사도 높은 순)
    return found_faqs

# --- LLM 서버 통신 관련 설정 ---
LLM_SERVER_URL = "http://localhost:8001/generate/" # FastAPI LLM 서버의 주소

# --- Django Views ---

def index(request):
    """메인 챗봇 UI 페이지를 렌더링합니다."""
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chat_api(request):
    """
    사용자의 챗봇 요청을 처리하고 LLM 답변을 스트리밍으로 반환하는 API 엔드포인트.
    FastAPI LLM 서버로부터 SSE 스트림을 받아 클라이언트로 다시 프록시합니다.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('question')
            service_category = data.get('category')

            if not user_question or not service_category:
                return JsonResponse({'error': '질문과 서비스 카테고리가 필요합니다.'}, status=400)

            print(f"사용자 질문: {user_question}, 카테고리: {service_category}")

            # 1. RAG 수행: 선택된 카테고리에서 관련 FAQ 검색
            relevant_faqs = search_faq_in_vector_db(user_question, service_category, top_k=1)

            # 2. LLM 프롬프트 구성 (컨텍스트 포함)
            context = ""
            if relevant_faqs:
                context += "다음은 사용자 질문과 관련된 FAQ 내용입니다:\\n"
                for i, faq_data in enumerate(relevant_faqs):
                    faq = faq_data['faq_object']
                    context += f"--- FAQ {i+1} ---\\n"
                    context += f"질문: {faq.question}\\n"
                    context += f"답변: {faq.answer}\\n"
                context += "--------------------\\n"
                context += "위 FAQ 내용 중 적절한 항목을 골라, '답변' 내용을 참고하여 사용자 질문에 답변해주세요. 만약 관련 정보가 없다면 '해당 서비스에 대한 문의 답변을 찾을 수 없습니다.'라고 답변하세요.\\n\\n"
            else:
                context += "관련 FAQ를 찾을 수 없습니다. 일반적인 지식으로 답변하거나, 관련 정보가 없다고 알려주세요.\\n\\n"

            context += "아래 질문에 대해 답하되, 줄바꿈('\\n') 및 기호을 적절히 활용하여 구조화된 답변을 제시하세요.\\n"
            llm_prompt = f"{context}사용자 질문: {user_question}"

            print(f"LLM으로 보낼 프롬프트:\\n{llm_prompt[:500]}...")

            # 3. LLM 서버 호출 및 스트리밍 응답 프록시
            def generate_response_stream(prompt):
                try:
                    # FastAPI LLM 서버에 스트리밍 요청
                    with requests.post(
                        LLM_SERVER_URL,
                        json={"prompt": prompt},
                        stream=True, # 스트리밍 응답을 받기 위해 True
                        timeout=120 # LLM 응답 대기 시간
                    ) as response:
                        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
                        for chunk in response.iter_content(chunk_size=None): # 청크 단위로 읽음
                            # FastAPI 서버는 이미 SSE 형식으로 데이터를 보내고 있으므로 그대로 yield
                            yield chunk # 받은 청크를 바로 클라이언트로 전달

                except requests.exceptions.ConnectionError:
                    yield f"data: {json.dumps('현재 AI 상담 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.')}\\n\\n"
                    yield "event: end\\ndata: \\n\\n"
                except requests.exceptions.Timeout:
                    yield f"data: {json.dumps('AI 상담 서버 응답 시간이 초과되었습니다. 다시 시도해주세요.')}\\n\\n"
                    yield "event: end\\ndata: \\n\\n"
                except requests.exceptions.RequestException as e:
                    yield f"data: {json.dumps(f'AI 상담 서버와의 통신 중 오류가 발생했습니다: {e}')}\\n\\n"
                    yield "event: end\\ndata: \\n\\n"
                except Exception as e:
                    yield f"data: {json.dumps(f'서버 오류가 발생했습니다: {e}')}\\n\\n"
                    yield "event: end\\ndata: \\n\\n"

            # StreamingHttpResponse로 제너레이터 반환
            return StreamingHttpResponse(generate_response_stream(llm_prompt), content_type="text/event-stream")

        except json.JSONDecodeError:
            return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            return JsonResponse({'error': f'서버 오류가 발생했습니다: {e}'}, status=500)
    else:
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)


