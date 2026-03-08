import logging
import re
from typing import Optional, Dict, Any, List

from openai import OpenAI

from .config import OPENAI_API_KEY
from .db import get_agent_config, get_conversation, upsert_conversation, get_memory, add_memory
from .php_client import get_next_slot, reserve_slot, get_entrevistador, send_whatsapp_message

logger = logging.getLogger(__name__)

# Respostas de fallback quando OpenAI falha
FALLBACK_MSG = "Desculpe, tive um problema técnico. Pode confirmar se deseja agendar uma entrevista? Responda 'sim' para continuarmos."


def _normalize_phone(phone: str) -> str:
    p = re.sub(r"\D", "", phone)
    if len(p) >= 10 and not p.startswith("55"):
        p = "55" + p
    return p


def _is_approval(text: str) -> bool:
    if not text or len(text) > 100:
        return False
    t = text.strip().lower()
    return any(
        x in t
        for x in (
            "sim",
            "pode ser",
            "confirmo",
            "confirmar",
            "pode",
            "ok",
            "beleza",
            "quero",
            "aceito",
        )
    )


def _apply_template(template: str, **kwargs) -> str:
    out = template
    for k, v in (kwargs or {}).items():
        out = out.replace("{{" + k + "}}", str(v or ""))
    return out


def run_agent(phone_id: str, first_name: str, text: str) -> str:
    """
    Processa a mensagem do usuário e retorna o texto de resposta.
    Usa OpenAI com tools (next_slot, reserve_slot, get_entrevistador) e memória no banco.
    """
    phone_id = _normalize_phone(phone_id)
    if not phone_id:
        return "Não foi possível identificar seu número. Por favor, tente novamente."

    config = get_agent_config()
    system_prompt = (
        (config and config.get("system_prompt"))
        or "Você é um assistente que agenda entrevistas por WhatsApp. Seja breve e cordial. Sugira sempre um único horário por vez (o mais próximo disponível). Quando o lead aprovar (sim, pode ser, confirmo), use a ferramenta reserve_slot e depois envie a confirmação com data, hora e nome do entrevistador."
    )
    template_suggestion = (
        (config and config.get("message_template_suggestion"))
        or "Legal {{firstName}}, podemos agendar uma entrevista às {{horaSug}}. Nesta entrevista online, que é bem rápida e não leva 10 minutos, irei explicar um pouco mais os detalhes da vaga e também darei algumas dicas para lograr êxito no processo seletivo. Pode ser {{horaSug}}?"
    )
    template_confirmation = (
        (config and config.get("message_template_confirmation"))
        or "Legal {{firstName}}, sua entrevista foi marcada e minutos antes iremos enviar o link.\nDia e horário: {{dataHora}}\nEntrevistador: {{nomeEntrevistador}}\n\nDesde já desejo boa sorte!"
    )

    conv = get_conversation(phone_id)
    flow_status = (conv and conv.get("flow_status")) or "start"
    current_slot_date = conv and conv.get("current_slot_date")
    current_slot_time = conv and conv.get("current_slot_time")
    current_responsible = conv and conv.get("current_responsible")
    candidate_name = (conv and conv.get("first_name")) or first_name or "Candidato(a)"

    # Fluxo direto: usuário aprovou o horário sugerido
    if flow_status == "waiting_slot_approval" and _is_approval(text) and current_slot_date and current_slot_time:
        result = reserve_slot(current_slot_date, current_slot_time, candidate_name)
        if result:
            schedule_id = result.get("schedule_id")
            entrevistador = result.get("entrevistador") or current_responsible or ""
            # Formata data/hora para exibição
            from datetime import datetime
            try:
                dt = datetime.strptime(f"{current_slot_date} {current_slot_time}", "%Y-%m-%d %H:%M")
                data_hora_str = dt.strftime("%d/%m/%Y às %H:%M")
            except Exception:
                data_hora_str = f"{current_slot_date} às {current_slot_time}"
            reply = _apply_template(
                template_confirmation,
                firstName=first_name or "Candidato(a)",
                dataHora=data_hora_str,
                nomeEntrevistador=entrevistador,
            )
            upsert_conversation(
                phone_id,
                first_name=first_name or conv.get("first_name"),
                flow_status="scheduled",
                schedule_id=schedule_id,
            )
            add_memory(phone_id, "user", text)
            add_memory(phone_id, "assistant", reply)
            return reply
        # Falha na reserva
        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", "Não consegui reservar esse horário agora. Deseja que eu sugira outro?")
        return "Não consegui reservar esse horário agora. Deseja que eu sugira outro?"

    # Obter próximo slot e sugerir
    slot = get_next_slot()
    if slot:
        hora_sug = slot.get("time") or slot.get("time", "")
        entrevistador = slot.get("entrevistador") or slot.get("responsible") or (config and config.get("default_entrevistador")) or ""
        upsert_conversation(
            phone_id,
            first_name=first_name or (conv and conv.get("first_name")),
            flow_status="waiting_slot_approval",
            current_slot_date=slot.get("date"),
            current_slot_time=slot.get("time"),
            current_responsible=entrevistador,
        )
        reply = _apply_template(
            template_suggestion,
            firstName=first_name or "Candidato(a)",
            horaSug=hora_sug,
        )
        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", reply)
        return reply

    # Sem slot disponível ou erro
    add_memory(phone_id, "user", text)
    fallback = "No momento não encontrei horários disponíveis. Tente novamente em breve ou entre em contato pelo outro canal."
    add_memory(phone_id, "assistant", fallback)
    return fallback


def run_agent_with_openai(phone_id: str, first_name: str, text: str) -> str:
    """
    Alternativa usando OpenAI com tool calling (pode ser usada no futuro para mais flexibilidade).
    Por agora run_agent() usa fluxo fixo + templates; esta função fica como opção.
    """
    if not OPENAI_API_KEY:
        return run_agent(phone_id, first_name, text)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        config = get_agent_config()
        model = (config and config.get("openai_model")) or "gpt-4o-mini"
        system_prompt = (config and config.get("system_prompt")) or (
            "Você é um assistente que agenda entrevistas por WhatsApp. Seja breve. "
            "Use a ferramenta get_next_slot para obter o próximo horário e sugira um por vez. "
            "Quando o usuário aprovar (sim, pode ser, confirmo), use reserve_slot e depois envie confirmação com data, hora e entrevistador."
        )

        memory = get_memory(phone_id, limit=16)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in memory:
            messages.append({"role": m["role"], "content": (m.get("content") or "")[:8000]})
        messages.append({"role": "user", "content": text})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_next_slot",
                    "description": "Obtém o próximo horário disponível para entrevista (um único slot).",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reserve_slot",
                    "description": "Reserva o horário para o candidato.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Data YYYY-MM-DD"},
                            "time": {"type": "string", "description": "Hora HH:MM"},
                            "candidate_name": {"type": "string", "description": "Nome do candidato"},
                        },
                        "required": ["date", "time", "candidate_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_entrevistador",
                    "description": "Retorna o nome do entrevistador padrão.",
                },
            },
        ]

        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = {}
                if tc.function.arguments:
                    import json
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        pass
                if name == "get_next_slot":
                    slot = get_next_slot()
                    result = slot or {"error": "Nenhum horário disponível"}
                elif name == "reserve_slot":
                    r = reserve_slot(
                        args.get("date", ""),
                        args.get("time", ""),
                        args.get("candidate_name", first_name or "Candidato(a)"),
                    )
                    result = r or {"error": "Falha ao reservar"}
                elif name == "get_entrevistador":
                    result = {"nome_entrevistador": get_entrevistador() or ""}
                else:
                    result = {}
                messages.append(choice.message)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
            # Segunda chamada para obter resposta final
            response2 = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
            final_content = response2.choices[0].message.content or FALLBACK_MSG
        else:
            final_content = choice.message.content or FALLBACK_MSG

        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", final_content)
        return final_content
    except Exception as e:
        logger.exception("OpenAI agent error: %s", e)
        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", FALLBACK_MSG)
        return FALLBACK_MSG
