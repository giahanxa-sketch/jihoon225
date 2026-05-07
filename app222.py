import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from datetime import datetime

# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(page_title="쇼핑몰 고객 응대 AI 챗봇", layout="wide")
st.title("🛒 쇼핑몰 전용 CS 상담 챗봇")
st.caption("고객님의 불편사항을 정중히 듣고 해결을 도와드리겠습니다.")

# --- 2. 환경 변수 및 API 키 설정 ---
# st.secrets에 키가 있으면 사용, 없으면 사이드바에서 입력을 받음
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.warning("API 키를 입력해주세요. (st.secrets 또는 사이드바)")
    st.stop()

genai.configure(api_key=api_key)

# --- 3. 설정 및 데이터 로드 (사이드바) ---
with st.sidebar:
    st.header("⚙️ 챗봇 설정")
    # 모델 선택 (lite 모델이 기본값)
    selected_model = st.selectbox(
        "모델 선택", 
        ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
        index=0
    )
    st.info(f"현재 사용 중인 모델: **{selected_model}**")
    
    # 대화 초기화 버튼
    if st.button("대화 내역 초기화"):
        st.session_state.messages = []
        st.rerun()

# --- 4. FAQ 데이터 로드 및 시스템 인스트럭션 구성 ---
faq_content = ""
if os.path.exists("faq_data.csv"):
    try:
        df_faq = pd.read_csv("faq_data.csv")
        faq_content = df_faq.to_markdown(index=False)
        has_faq = True
    except Exception as e:
        st.error(f"CSV 파일을 읽는 중 오류 발생: {e}")
        has_faq = False
else:
    has_faq = False

# 시스템 프롬프트 작성
system_instruction = """
1. 당신은 쇼핑몰의 전문 고객 상담사입니다. 사용자의 불편/불만에 대해 정중하고 공감 어린 말투로 응답하세요.
3. 사용자의 불편 사항을 구체적(무엇이/언제/어디서/어떻게)으로 정리하여 수집하고, 이를 사내 고객 응대 담당자에게 전달한다는 취지를 안내하세요.
4. 대화의 마지막 단계에서는 담당자가 확인 후 회신할 수 있도록 사용자의 이메일 주소를 요청하세요. 
   만약 사용자가 연락처 제공을 거부하면: "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 직접 안내해 드리기 어렵습니다."라고 정중히 마무리하세요.
"""

if has_faq:
    system_instruction += f"""
2. 답변을 할 때는 제공된 [CSV 참조 데이터]를 우선적으로 확인하여 안내하세요.
   데이터에 없는 내용이라면 임의로 지어내지 말고 "담당 부서 확인 후 안내해 드리겠습니다"라고 답변하세요.
   
[CSV 참조 데이터]
{faq_content}
"""

# --- 5. 세션 상태 관리 (대화 히스토리) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. 챗봇 UI 및 대화 출력 ---
# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력창
if prompt := st.chat_input("불편하신 점을 말씀해 주세요."):
    # 1. 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Gemini API 호출
    try:
        model = genai.GenerativeModel(
            model_name=selected_model,
            system_instruction=system_instruction
        )
        
        # 최근 6턴(12개 메시지)만 히스토리로 전달하여 토큰 최적화
        history_to_send = []
        for m in st.session_state.messages[-12:]:
            history_to_send.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
        
        # 채팅 시작 (마지막 입력 제외한 히스토리 전달)
        chat = model.start_chat(history=history_to_send[:-1])
        
        with st.chat_message("assistant"):
            response = chat.send_message(prompt)
            st.markdown(response.text)
            # 3. 모델 메시지 저장
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        # 에러 핸들링 (429 ResourceExhausted 등)
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            st.error("현재 사용량이 많아 응답이 지연되고 있습니다. 1분 뒤에 다시 시도해 주세요.")
        else:
            st.error(f"오류가 발생했습니다: {error_msg}")

# --- 7. 로그 기록 및 다운로드 기능 ---
if st.session_state.messages:
    st.divider()
    # 대화 내역을 DataFrame으로 변환
    log_df = pd.DataFrame(st.session_state.messages)
    log_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CSV 다운로드 버튼
    csv_log = log_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 전체 대화 내역 다운로드 (CSV)",
        data=csv_log,
        file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
