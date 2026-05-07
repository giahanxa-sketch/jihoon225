import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import io

# -----------------------------------------------------------------------------
# 1. 초기 설정 및 데이터 로드 (선택적 CSV 로드)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="고객 응대 AI 챗봇", page_icon="🛍️")
st.title("🛍️ 쇼핑몰 고객 지원 챗봇")

FAQ_FILE = "faq_data.csv"
faq_context = ""

# 파일이 존재할 경우에만 데이터를 읽어옴
if os.path.exists(FAQ_FILE):
    try:
        df = pd.read_csv(FAQ_FILE)
        faq_context = df.to_markdown(index=False)
        st.success("✅ 사용자 맞춤형 FAQ 데이터가 성공적으로 적용되었습니다.") # 학생들을 위한 시각적 피드백
    except Exception as e:
        st.warning(f"⚠️ CSV 파일을 읽는 중 오류가 발생했습니다. (기본 모드로 작동합니다): {e}")
else:
    st.info("💡 'faq_data.csv' 파일을 업로드하면 챗봇이 해당 데이터를 참고하여 답변합니다.")

# -----------------------------------------------------------------------------
# 2. 사이드바 UI 및 환경 변수 설정 (Sidebar & Config)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 챗봇 설정")
    
    # 모델 선택 (실험 버전 제외)
    model_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    selected_model = st.selectbox("기본 모델 선택", model_options)
    
    # API 키 입력 처리 (st.secrets 우선 확인 후 없으면 UI 입력)
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API Key를 입력하세요", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.warning("API 키를 입력해야 챗봇이 작동합니다.")

    # 편의 기능: 대화 초기화 버튼
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"현재 모델: `{selected_model}`")

# -----------------------------------------------------------------------------
# 3. 시스템 프롬프트 구성 (조건부 데이터 주입)
# -----------------------------------------------------------------------------
system_prompt = """
당신은 쇼핑몰의 전문 고객 상담사입니다. 사용자의 불편/불만에 대해 정중하고 공감 어린 말투로 응답하세요.

[행동 지침]
1. 사용자의 불편 사항을 구체적(무엇이/언제/어디서/어떻게)으로 정리하여 수집하고, 이를 사내 담당자에게 전달한다는 취지를 안내하세요.
2. 대화의 마지막 단계에서는 담당자가 확인 후 회신할 수 있도록 사용자의 이메일 주소를 요청하세요. 
3. 만약 사용자가 연락처 제공을 거부하면: "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 직접 안내해 드리기 어렵습니다."라고 정중히 마무리하세요.
"""

# csv 파일 데이터가 성공적으로 로드된 경우에만 프롬프트에 추가 규칙 부여
if faq_context:
    system_prompt += f"""
4. 답변 시 아래 제공된 [CSV 참조 데이터]를 우선적으로 확인하여 규정과 절차에 맞게 안내하세요. 
5. 데이터에 없는 내용이라면 임의로 지어내지(Hallucination) 말고, "담당 부서 확인 후 안내해 드리겠습니다"라고 답변하세요.

[CSV 참조 데이터]
{faq_context}
"""

# -----------------------------------------------------------------------------
# 4. 상태 관리 (Session State Management)
# -----------------------------------------------------------------------------
# Streamlit은 화면이 다시 그려질 때마다 변수가 초기화되므로, session_state에 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 화면에 이전 대화 내역 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------------------------------------------
# 5. 모델 API 호출 및 예외 처리 (API Call & Exception Handling)
# -----------------------------------------------------------------------------
def get_gemini_response(chat_session, user_prompt):
    # 재시도(tenacity) 기능 제거: 에러가 나면 억지로 재시도하지 않고 바로 보여주기 위함
    return chat_session.send_message(user_prompt)

# -----------------------------------------------------------------------------
# 6. 메인 채팅 로직 (Chat Interaction)
# -----------------------------------------------------------------------------
if prompt := st.chat_input("불편하신 사항을 말씀해 주세요."):
    if not api_key:
        st.error("좌측 사이드바에 API 키를 먼저 입력해 주세요.")
    else:
        # 1) 사용자 메시지 화면에 출력 및 상태 저장
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2) 메모리 관리: 최근 6턴(User+Model 왕복 12개 메시지)만 유지하여 토큰 최적화
        recent_messages = st.session_state.messages[-12:]
        
        # Gemini API 형식에 맞게 히스토리 변환 ('user' 또는 'model')
        formatted_history = []
        for msg in recent_messages[:-1]: # 마지막 사용자 입력 제외한 히스토리
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})

        # 3) 모델 초기화 (시스템 프롬프트 주입) 및 채팅 세션 시작
        model = genai.GenerativeModel(
            model_name=selected_model,
            system_instruction=system_prompt
        )
        chat = model.start_chat(history=formatted_history)

        # 4) AI 응답 생성 및 화면 출력
        with st.chat_message("assistant"):
            with st.spinner("답변을 작성하고 있습니다..."):
                try:
                    response = get_gemini_response(chat, prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error("현재 일시적인 통신 지연이 발생했습니다. 잠시 후 다시 시도해 주세요.")
                    st.error(f"상세 에러: {e}")

# -----------------------------------------------------------------------------
# 7. 로그 다운로드 기능 (Log Download)
# -----------------------------------------------------------------------------
if len(st.session_state.messages) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 대화 로그 다운로드")
    
    # 대화 기록을 pandas DataFrame으로 변환 후 CSV로 추출
    log_df = pd.DataFrame(st.session_state.messages)
    csv_buffer = log_df.to_csv(index=False).encode('utf-8-sig') # 한글 깨짐 방지 인코딩
    
    st.sidebar.download_button(
        label="대화 기록 다운로드 (CSV)",
        data=csv_buffer,
        file_name="chat_history_log.csv",
        mime="text/csv"
    )


# ----- OLD VERSION -----
# import streamlit as st
# import google.generativeai as genai
# import os, re, time, uuid, csv, datetime
# from pathlib import Path

# st.set_page_config(page_title="고객 응대 챗봇", page_icon="🛍️")
# st.title("고객 응대 챗봇 (Gemini + Streamlit)")
# st.caption("정중 응대 · 불편 수집 · 담당자 전달 · 이메일 수집")

# # -----------------------------
# # 0) 공통 유틸
# # -----------------------------
# def today_str():
#     return datetime.datetime.now().strftime("%Y-%m-%d")

# def now_iso():
#     return datetime.datetime.now().isoformat(timespec="seconds")

# # 세션 ID (한 번 생성 후 유지)
# if "session_id" not in st.session_state:
#     st.session_state.session_id = uuid.uuid4().hex[:10]

# # -----------------------------
# # 1) API 키
# # -----------------------------
# API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
# if not API_KEY:
#     with st.expander("🔐 GEMINI_API_KEY가 없나요? 여기를 눌러 임시 입력"):
#         API_KEY = st.text_input("Gemini API 키", type="password")
#     if not API_KEY:
#         st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets에 추가하세요.")
#         st.stop()

# genai.configure(api_key=API_KEY)
# st.sidebar.write(f"google-generativeai 버전: **{genai.__version__}**")

# # -----------------------------
# # 2) 사용가능 모델 조회 + 기본값을 2.0-flash로
# # -----------------------------
# try:
#     raw_models = list(genai.list_models())
#     avail = [m for m in raw_models if "generateContent" in getattr(m, "supported_generation_methods", [])]
#     names = [m.name.replace("models/", "") for m in avail]
# except Exception as e:
#     st.error(f"모델 목록 조회 실패: {e}")
#     st.stop()

# # 실습에선 2.0/2.5 중 '비-실험(-exp 없는)' 모델만 사용
# def is_safe(n: str) -> bool:
#     if "-exp" in n:     # 실험 모델 제외
#         return False
#     return bool(re.match(r"^gemini-(2\.0|2\.5)-", n))

# safe = [n for n in names if is_safe(n)]

# # 선호 순서: 2.0-flash → 2.5-flash → 2.0-pro → 2.5-pro
# PREF = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-2.5-pro"]

# def pick_default():
#     # 1) 선호 목록에서 첫 매칭
#     for want in PREF:
#         if want in safe:
#             return want
#     # 2) 그래도 없으면 safe의 첫 번째나 names 첫 번째
#     return safe[0] if safe else (names[0] if names else None)

# default_model = pick_default()
# if not default_model:
#     st.error("사용 가능한 generateContent 모델을 찾지 못했습니다. 키/권한/리전을 확인하세요.")
#     st.stop()

# # 사이드바: 모델 선택(기본값을 gemini-2.0-flash로 세팅)
# opts = safe if safe else names
# default_index = opts.index(default_model)
# model_name = st.sidebar.selectbox("사용할 모델", options=opts, index=default_index)

# # -----------------------------
# # 3) 시스템 프롬프트
# # -----------------------------
# SYSTEM_PROMPT = """
# 당신은 아래 기준에 따라 답변하는 고객 응대용 AI 챗봇입니다.

# --- [참고 기준 시작] ---
# 1) 사용자는 쇼핑몰 구매 과정에서 겪은 불편/불만을 언급합니다. 정중하고 공감 어린 말투로 응답하세요.
# 2) 사용자의 불편 사항을 구체적으로 정리하여(무엇이/언제/어디서/어떻게) 수집하고, 이를 고객 응대 담당자에게 전달한다는 취지로 안내하세요.
# 3) 마지막에는 담당자 확인 후 회신을 위해 이메일 주소를 요청하세요.
#    - 사용자가 연락 제공을 원치 않으면:
#      "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요."라고 정중히 고지하세요.
# --- [참고 기준 끝] ---

# 추측하거나 사실이 아닌 내용은 말하지 마세요.
# """

# # -----------------------------
# # 4) 모델/세션 초기화
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def get_model(name: str):
#     return genai.GenerativeModel(model_name=name, system_instruction=SYSTEM_PROMPT)

# model = get_model(model_name)

# if "chat" not in st.session_state:
#     st.session_state.chat = model.start_chat(history=[])
# if "messages" not in st.session_state:
#     st.session_state.messages = []  # [(role, text)]

# st.success(f"선택된 모델: **{model_name}**  | 세션ID: `{st.session_state.session_id}`")

# # -----------------------------
# # 5) 대화 자동 기록 옵션 (CSV)
# # -----------------------------
# st.sidebar.markdown("### 📝 자동 기록")
# save_enabled = st.sidebar.checkbox("대화 자동 기록 (CSV)", value=False,
#                                    help="체크하면 로그 폴더에 CSV로 자동 저장됩니다.")
# log_dir = Path("logs")
# log_dir.mkdir(exist_ok=True)
# log_path = log_dir / f"chat_{today_str()}.csv"

# def append_log(role: str, text: str):
#     if not save_enabled:
#         return
#     # CSV 헤더: ts, session_id, model, role, text
#     new_file = not log_path.exists()
#     with open(log_path, "a", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         if new_file:
#             writer.writerow(["timestamp", "session_id", "model", "role", "text"])
#         writer.writerow([now_iso(), st.session_state.session_id, model_name, role, text])

# # 로그 파일 다운로드 버튼
# if log_path.exists():
#     with open(log_path, "rb") as f:
#         st.sidebar.download_button("📥 오늘 로그 다운로드", f, file_name=log_path.name)

# # -----------------------------
# # 6) 이전 대화 표시
# # -----------------------------
# for role, text in st.session_state.messages:
#     with st.chat_message("ai" if role == "ai" else "user"):
#         st.markdown(text)

# # -----------------------------
# # 7) 안전하게 전송 (429 방어 포함)
# # -----------------------------
# def send_safely(msg: str):
#     try:
#         return st.session_state.chat.send_message(msg)
#     except Exception as e:
#         s = str(e)
#         if "429" in s:
#             # 최근 6턴만 유지하고 잠깐 대기 후 재시도
#             trimmed = st.session_state.chat.history[-6:]
#             st.session_state.chat = model.start_chat(history=trimmed)
#             time.sleep(2)
#             return st.session_state.chat.send_message(msg)
#         raise

# # -----------------------------
# # 8) 입력/응답 & 자동 기록
# # -----------------------------
# if prompt := st.chat_input("불편/요청 사항을 입력하세요"):
#     st.session_state.messages.append(("user", prompt))
#     append_log("user", prompt)

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("ai"):
#         with st.spinner("답변 생성 중..."):
#             try:
#                 resp = send_safely(prompt)
#                 bot_text = resp.text
#                 st.markdown(bot_text)
#                 st.session_state.messages.append(("ai", bot_text))
#                 append_log("ai", bot_text)
#             except Exception as e:
#                 st.error(f"오류: {e}")

# # -----------------------------
# # 9) 도구: 대화 초기화
# # -----------------------------
# cols = st.columns(2)
# with cols[0]:
#     if st.button("🧹 대화 초기화"):
#         st.session_state.messages = []
#         st.session_state.chat = model.start_chat(history=[])
#         st.rerun()
# with cols[1]:
#     st.caption("TIP: 이메일 주소는 마지막에 꼭 남겨 주세요.")
