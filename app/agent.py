import logging
import re
from typing import Dict, List

from openai import OpenAI

from .config import OPENAI_API_KEY
from .db import get_agent_config, get_memory, add_memory
from .php_client import get_next_slot, reserve_slot, get_entrevistador

logger = logging.getLogger(__name__)

# Respostas de fallback quando OpenAI falha
FALLBACK_MSG = "Desculpe, tive um problema técnico. Pode confirmar se deseja agendar uma entrevista? Responda 'sim' para continuarmos."


def _normalize_phone(phone: str) -> str:
    p = re.sub(r"\D", "", phone)
    if len(p) >= 10 and not p.startswith("55"):
        p = "55" + p
    return p


def run_agent_with_openai(phone_id: str, first_name: str, text: str) -> str:
    """
    Agente conversacional via OpenAI com tools (get_next_slot, reserve_slot).
    Usa prompt e temperatura configuráveis no sistema.
    """
    phone_id = _normalize_phone(phone_id)
    if not phone_id:
        return "Não foi possível identificar seu número. Por favor, tente novamente."

    config = get_agent_config()
    api_key = (config and config.get("openai_api_key")) or OPENAI_API_KEY
    if not api_key:
        return "Desculpe, a integração com IA não está configurada. Entre em contato pelo outro canal."

    try:
        client = OpenAI(api_key=api_key)
        model = (config and config.get("openai_model")) or "gpt-4o-mini"
        temperature = float((config and config.get("temperature")) or 0.7)
        temperature = min(2.0, max(0.0, temperature))
        base_prompt = (config and config.get("system_prompt")) or (
            "Você é um assistente cordial que agenda entrevistas por WhatsApp. "
            "Converse naturalmente. Use get_next_slot para obter o próximo horário disponível e sugira UM por vez, sempre informando a DATA e a HORA (ex: 'dia 15/03 às 08:00'). "
            "Se o candidato perguntar 'qual dia?' ou 'de qual dia?', responda com a data do horário que você sugeriu. "
            "Quando o candidato aprovar (sim, pode ser, confirmo, ok, beleza), use reserve_slot com a data (YYYY-MM-DD) e hora (HH:MM) do slot que você sugeriu, e depois envie a confirmação com data, hora e nome do entrevistador. "
            "Seja breve e objetivo nas mensagens."
        )
        candidate_name = first_name or "Candidato(a)"
        system_prompt = f"{base_prompt}\n\nO nome do candidato nesta conversa é: {candidate_name}."

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
                    "description": "Obtém o próximo horário disponível para entrevista. Retorna date (YYYY-MM-DD), time (HH:MM) e entrevistador. Use sempre que for sugerir um horário.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reserve_slot",
                    "description": "Reserva o horário para o candidato. Use APENAS quando o candidato aprovar (sim, pode ser, confirmo). Passe a data e hora EXATAS do slot que você sugeriu.",
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

        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=temperature)
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
            response2 = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=temperature)
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
