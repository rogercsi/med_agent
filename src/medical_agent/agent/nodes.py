from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from medical_agent.agent.state import PatientMemory
from medical_agent.config import get_settings
from medical_agent.memory.mem0_client import add_memory, search_memory
from medical_agent.rag.pipeline import get_rag_pipeline

_EMERGENCY_KEYWORDS = [
    "压迫性胸痛", "胸痛", "放射至左臂", "向左臂", "左臂疼痛",
    "下颌疼痛", "濒死感", "意识丧失", "失去意识", "晕厥",
    "大出血", "呼吸困难", "无法呼吸", "心脏骤停",
    "突然剧烈头痛", "半身不遂", "口角歪斜", "中风",
    "严重过敏反应", "血压骤降", "休克",
]


def _make_llm() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _get_last_user_message(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


async def node_inject_memory(state: dict) -> dict:
    query = _get_last_user_message(state)
    patient_id = state.get("patient_id", "anonymous")

    memories = search_memory(query, patient_id, limit=10)

    pm = PatientMemory()
    for m in memories:
        text = m.get("memory", "")
        if any(k in text for k in ["过敏", "不耐受"]):
            pm.allergies.append(text)
        elif any(k in text for k in ["糖尿病", "高血压", "冠心病", "哮喘", "慢性"]):
            pm.chronic_conditions.append(text)
        elif any(k in text for k in ["服用", "用药", "片", "mg", "剂量"]):
            pm.current_medications.append(text)
        elif any(k in text for k in ["症状", "疼痛", "咳嗽", "发热", "头痛"]):
            pm.symptoms_history.append(text)
        elif any(k in text for k in ["偏好", "不愿", "希望", "建议"]):
            pm.preferences.append(text)

    return {
        "patient_memory": pm,
        "current_phase": "memory_inject",
    }


async def node_intake(state: dict) -> dict:
    settings = get_settings()
    llm = _make_llm()

    user_query = _get_last_user_message(state)
    pm: PatientMemory = state.get("patient_memory", PatientMemory())
    summary = state.get("conversation_summary", "")

    system = (
        "你是一名专业的医疗助手，负责理解患者的症状描述。\n"
        f"患者历史信息：\n{pm.to_prompt_str()}\n"
        + (f"\n对话摘要：{summary}" if summary else "")
    )

    resp = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"患者描述：{user_query}\n请简洁地提炼关键症状和主诉（50字内）："},
        ],
        temperature=0,
        max_tokens=100,
    )
    refined = (resp.choices[0].message.content or user_query).strip()

    return {
        "rewritten_query": refined,
        "current_phase": "intake",
        "turn_count": state.get("turn_count", 0) + 1,
    }


async def node_safety_check(state: dict) -> dict:
    user_query = _get_last_user_message(state)
    combined = user_query + " " + state.get("rewritten_query", "")

    triggered = [kw for kw in _EMERGENCY_KEYWORDS if kw in combined]
    is_emergency = len(triggered) >= 2 or any(
        kw in combined for kw in ["意识丧失", "心脏骤停", "大出血", "休克"]
    )

    return {
        "is_emergency": is_emergency,
        "emergency_keywords": triggered,
        "current_phase": "safety_check",
    }


async def node_retrieve_context(state: dict) -> dict:
    query = state.get("rewritten_query") or _get_last_user_message(state)
    summary = state.get("conversation_summary", "")

    pipeline = get_rag_pipeline()
    result = pipeline.query(query, mode="optimized", history_summary=summary)

    return {
        "retrieved_chunks": result.chunks,
        "retrieved_sources": result.sources,
        "rewritten_query": result.rewritten_query,
        "token_before_compress": result.token_before_compress,
        "token_after_compress": result.token_after_compress,
        "current_phase": "retrieve",
    }


async def node_generate_response(state: dict) -> dict:
    settings = get_settings()
    llm = _make_llm()

    pm: PatientMemory = state.get("patient_memory", PatientMemory())
    chunks: list[str] = state.get("retrieved_chunks", [])
    query = _get_last_user_message(state)
    summary = state.get("conversation_summary", "")
    is_emergency: bool = state.get("is_emergency", False)

    if is_emergency:
        answer = (
            "⚠️ **紧急提示**：您描述的症状可能提示急性心肌梗死或其他危及生命的情况。\n\n"
            "**请立即拨打120！不要自行驾车就医！**\n\n"
            "等待急救期间：\n"
            "1. 立即停止活动，平卧休息\n"
            "2. 如无阿司匹林过敏，可嚼服阿司匹林300 mg\n"
            "3. 保持气道通畅，解开领口\n"
            "4. 保持镇静，等待急救人员\n\n"
            "时间就是生命，请立即行动！"
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "current_phase": "respond",
        }

    context_str = "\n\n---\n\n".join(chunks) if chunks else "暂无相关医疗知识"
    system = (
        "你是一名专业、耐心的医疗助手，基于提供的医疗知识为患者提供参考信息。\n"
        "重要原则：\n"
        "1. 仅基于提供的医疗知识回答，不臆测诊断\n"
        "2. 建议患者就医而非自行诊断\n"
        "3. 回答简洁清晰，使用患者能理解的语言\n"
        "4. 如有过敏或禁忌证，必须明确提示\n\n"
        f"患者历史信息：\n{pm.to_prompt_str()}\n"
        + (f"\n对话摘要：{summary}" if summary else "")
        + f"\n\n医疗知识参考：\n{context_str}"
    )

    resp = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    answer = resp.choices[0].message.content or "抱歉，无法生成回复，请稍后再试。"

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_phase": "respond",
    }


async def node_save_memory(state: dict) -> dict:
    patient_id = state.get("patient_id", "anonymous")
    messages = state.get("messages", [])

    turn_messages = []
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            turn_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            turn_messages.append({"role": "assistant", "content": msg.content})

    if turn_messages:
        try:
            add_memory(turn_messages, patient_id)
        except Exception:
            pass  # memory save failure should not break the main flow

    return {"current_phase": "save_memory"}


async def node_summarize_conversation(state: dict) -> dict:
    settings = get_settings()
    llm = _make_llm()

    messages = state.get("messages", [])
    existing_summary = state.get("conversation_summary", "")

    # Take last 6 turns to summarize
    recent = []
    for msg in messages[-12:]:
        if isinstance(msg, HumanMessage):
            recent.append(f"患者：{msg.content}")
        elif isinstance(msg, AIMessage):
            recent.append(f"助手：{msg.content[:200]}")

    if not recent:
        return {}

    history_text = "\n".join(recent)
    prompt = (
        f"请将以下对话摘要（如有）和最新对话内容合并成一个简洁摘要（100字内），"
        f"重点保留医疗相关信息：\n\n"
        + (f"现有摘要：{existing_summary}\n\n" if existing_summary else "")
        + f"最新对话：\n{history_text}"
    )

    resp = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    new_summary = (resp.choices[0].message.content or existing_summary).strip()

    return {
        "conversation_summary": new_summary,
        "current_phase": "summarize",
    }
