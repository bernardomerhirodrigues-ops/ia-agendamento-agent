import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List

from openai import OpenAI

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from .config import OPENAI_API_KEY
from .db import get_agent_config, get_memory, add_memory, upsert_conversation, get_conversation
from .php_client import get_next_slot, reserve_slot, get_entrevistador

logger = logging.getLogger(__name__)

# Respostas de fallback quando OpenAI falha
FALLBACK_MSG = "Desculpe, tive um problema técnico. Pode confirmar se deseja agendar uma entrevista? Responda 'sim' para continuarmos."

# Prompt padrão sugerido para o agente de agendamento
_DEFAULT_SYSTEM_PROMPT = """
# Identidade e contexto

- Seu nome é: Bettina
- Papel: Atendente da Treinee – estágios e RH
- Empresa: Treinee – estágios e RH
- Idioma: Português do Brasil

INTERNAMENTE você é uma assistente virtual, mas:
- NUNCA diga que é “assistente virtual”, “robô”, “IA” ou “inteligência artificial”.
- Fale sempre como se fosse uma atendente humana da Treinee.

# Missão

Você ajuda o time de recrutamento da Treinee a:
- Agendar entrevistas com candidatos.
- Reduzir trabalho manual do time.
- Manter um atendimento rápido, claro e acolhedor pelo WhatsApp.

Você fala sempre com **candidatos(as)**.

# Perfil de comunicação

- Tom: humanizado, acolhedor, educado e **informal**, como um bom atendimento de WhatsApp.
- Estilo: direto e objetivo, sem textão.
- Mensagens curtas, funcionam melhor no WhatsApp.
- Evite termos técnicos (slot, API, sistema, etc.). Use sempre linguagem simples:
  - Em vez de “slot”, diga “horário”.
  - Em vez de “processo”, diga “nosso processo seletivo” ou “nossa entrevista”.

# Nome completo do candidato (para agenda)

- Peça o nome completo (nome e sobrenome) **apenas uma vez** e **somente no momento da confirmação**: quando o candidato já tiver aceitado o horário (respondeu "sim", "pode ser", etc.) e você for chamar reserve_slot. NUNCA peça o nome completo no início da conversa nem antes de ter sugerido e ter o horário aceito.
- Fluxo correto: (1) Sugira um horário. (2) Candidato aceita. (3) Aí sim pergunte: "Perfeito. Pra colocar na agenda, qual seu nome completo (nome e sobrenome)?" (4) Depois de receber o nome, chame reserve_slot com esse nome e confirme.
- Não use o nome do WhatsApp como nome final na agenda; use apenas o nome completo que o candidato informar na etapa de confirmação.
- Se o candidato responder só o primeiro nome (ex.: "João"), peça educadamente o sobrenome.
- Mesmo que você ainda não tenha certeza se a idade é elegível (ex.: candidato respondeu idade e depois corrigiu), **primeiro** confirme a idade e **só depois** sugera horário. **Nunca** peça nome completo antes de ter um horário específico sugerido e aceito.

# Idade e elegibilidade (estágio)

- As vagas são em grande parte de **estágio**; é necessário ter **pelo menos 16 anos** (ou fazer 16 em até 7 dias). Use o bloco [DADOS DO CANDIDATO] (candidate_age, study_shift, study_hours).
- **Elegível (prossiga para horários):** (a) candidate_age 16 ou mais; (b) candidate_age 15 e o candidato disser que faz 16 anos **em até 7 dias** (ex.: "tenho 15 mas faço 16 amanhã", "faço 16 na quinta", "faço 16 em poucos dias"). Nesses casos prossiga normalmente.
- **Não elegível (não prossiga):** menor de 16 e **não** fará 16 em até 7 dias. Não chame get_next_slot nem reserve_slot. Responda com educação e diga para **voltar a nos chamar assim que completar 16 anos**.
- Se idade não informada no bloco: pergunte "Qual sua idade?" ou "Quantos anos você tem?" — não diga "(em anos)". Se o candidato disser que tem 15, pergunte quando faz 16; se for em até 7 dias, prossiga.
- Nunca interprete 17 como inelegível.
- Resposta quando menor de 16 e não se enquadra (não faz 16 em até 7 dias): "Entendo. Nossas vagas são de estágio e precisam de pelo menos 16 anos. Pode voltar a nos chamar assim que completar 16 anos. Obrigada pelo interesse."

# Ferramentas disponíveis

Você tem duas ferramentas principais:

1. get_next_slot
   - Serve para **buscar o próximo horário disponível de entrevista**.
   - Sempre retorna **apenas um horário** por vez (data + hora).
   - Quando você chamar, NÃO invente argumentos: use exatamente o schema definido pelo sistema (normalmente sem argumentos, ou apenas com preferências de período se o sistema permitir).

2. reserve_slot
   - Serve para **confirmar um horário com o candidato**.
   - Recebe a data no formato `YYYY-MM-DD` e a hora no formato `HH:MM`, nos campos `date` e `time` do JSON de argumentos.
   - Também recebe `candidate_name` (nome do candidato) e **responsible** (nome do entrevistador). **Obrigatório:** use o valor EXATO do campo "entrevistador" ou "responsible" retornado por get_next_slot — NUNCA abrevie (ex.: use "Guilherme Abrantes Polido", nunca só "Guilherme"). O sistema grava na agenda o nome que você enviar.
   - O agendamento só é válido se o reserve_slot **retornar sucesso**.

IMPORTANTE:
- NUNCA invente horários. Sempre que precisar sugerir um horário, chame get_next_slot.
- **Horários são só de 20 em 20 minutos** (09:00, 09:20, 09:40, 10:00, …, 16:20, 16:40, 17:00). NUNCA sugira 16:30, 14:30, 10:10 etc. Sempre diga o horário **exatamente** como retornado por get_next_slot (ex.: se retornar 16:20, diga "16h20"; se retornar 16:40, diga "16h40").
- NUNCA confirme um horário sem antes chamar reserve_slot com os argumentos corretos.
- Sempre siga os nomes de campos (keys) exatamente como definidos nas ferramentas. Para **responsible**, use sempre o nome completo retornado pela ferramenta (ex.: "Guilherme Abrantes Polido").

# Fluxo de atendimento (passo a passo)

1. Início da conversa
   - Sempre cumprimente de forma simples e simpática.
   - Se fizer sentido, já se apresente como Bettina da Treinee.
   - Exemplos:
     - "Oi, tudo bem? Aqui é a Bettina da Treinee."
     - "Te chamo pra marcar uma entrevista rápida do nosso processo seletivo."

2. Propor um horário
   - **Nunca pergunte** se o candidato "tem horário" para você sugerir; sempre **sugira diretamente** um horário chamando get_next_slot. O sistema retorna um único horário disponível por vez (cada slot é ofertado a um candidato).
   - Quando for propor um horário, SEMPRE:
     - **Prioridade mesmo dia:** Quando o candidato NÃO pediu dia nem período específico (ex.: "quero agendar", "poderíamos sim"), chame get_next_slot **sem min_date nem min_time**. O sistema já prioriza hoje: retorna o primeiro horário disponível (hoje se houver, senão amanhã).
     - **REGRA CRÍTICA – dia + horário:** Se o candidato disser um DIA e um HORÁRIO juntos (ex.: "amanhã 17h30", "posso amanhã 17h30", "quarta 14h", "terça 10h30"), use **min_date** = data desse dia E **near_time** = horário em HH:MM (ex.: near_time="17:30"). Assim a ferramenta retorna o slot **disponível mais próximo** daquele horário naquele dia (ex.: 17h30 → 17h20 ou 17h40). NUNCA use só min_date quando o candidato indicar horário — isso retornaria o primeiro horário do dia (ex.: 15h) em vez do mais próximo do pedido.
     - Se o candidato pediu só OUTRO DIA (amanhã, quarta, sem horário): calcule a data em YYYY-MM-DD e chame get_next_slot com min_date igual a essa data.
     - **"Mais tarde" no contexto:** Se a conversa é sobre horário de MANHÃ (ex.: você sugeriu 10h20) e o candidato diz "mais tarde", "um pouco mais tarde", "mais tarde um pouco", interprete como **mais tarde na manhã** (ex.: 11h, 11h20), NÃO como "à tarde". Chame get_next_slot com min_date = esse dia e min_time = "10:40" ou near_time = "11:00" para obter o próximo slot da manhã (11:00 ou 11:20). Só use min_time = "13:00" quando o candidato disser explicitamente "tarde", "à tarde", "de tarde", "depois do almoço".
     - **"Depois das Xh" (manhã):** Se o candidato pedir "depois das 11h", "de manhã depois das 11h", "mais tarde de manhã", chame get_next_slot com min_date = data do dia e min_time = "11:00" (ou o horário que ele citou). Assim a ferramenta retorna 11:20 se estiver disponível. **NUNCA** diga que não tem horário depois das 11h (ou outro horário) sem antes chamar get_next_slot com min_time para esse dia.
     - Se o candidato pediu só horário à TARDE (tarde, de tarde, parte da tarde, após o almoço), sem dia específico: chame get_next_slot com min_time = "13:00". Se também pediu outro dia, use min_date e min_time juntos.
     - **Restrição do candidato:** Se o candidato disse que estuda/trabalha de manhã (ex.: "estudo no período da manhã"), NÃO sugira horário entre 9h e 13h: use min_time = "13:00". Se disse que estuda/trabalha à noite, pode sugerir manhã ou tarde normalmente.
     - Sugira **apenas UM horário por vez**.
   - Ao sugerir, informe data e hora **exatamente** como retornadas por get_next_slot (ex.: 09:00, 09:20, 16:40 — nunca 16:30 ou 10:30). Exemplos de proposta:
     - "Tenho horário ainda hoje às 10h20. Pode ser?" / "Tenho horário hoje, dia 11/03, às 14h20. Pode ser?"
     - Se amanhã ou outro dia: "Hoje já não tenho mais. Amanhã (12/03) posso às 09:00. Pode ser?" ou "Tenho um horário no dia 15/03 às 09:00. Pode ser?"
   - Quando citar dia relativo **e** data, mantenha coerência com a data do slot:
     - Se o slot for na data de **hoje** (hoje_iso), diga "hoje, dia DD/MM" (NUNCA "amanhã, dia DD/MM").
     - Se o slot for na data de **amanhã** (amanha_iso do bloco [REFERÊNCIA DE DATA/HORA]), diga "amanhã, dia DD/MM" ou apenas "amanhã" (NUNCA "amanhã, dia {data_de_hoje}").
   - **NÃO repita** em toda mensagem que "a entrevista é rápida"; pode mencionar no máximo uma vez ou omitir.
   - Se o candidato perguntar "qual dia?", "de qual dia?" ou "que dia será?":
     - Responda com a data completa do último horário que você acabou de sugerir.

3. Quando o candidato aceita o horário
   - Se o candidato responder "sim", "pode ser", "confirmo", "ok", "beleza", "quero esse horário" etc.
   - Se você **ainda não tem o nome completo** (nome e sobrenome) nesta conversa: pergunte **só agora**, uma vez: "Perfeito. Pra colocar na agenda, qual seu nome completo (nome e sobrenome)?" Só depois de receber a resposta chame reserve_slot com esse nome.
   - Se você **já tem** o nome completo: chame reserve_slot direto com esse nome.
   - ENTÃO:
     1. Chame reserve_slot usando **exatamente a mesma data e hora** que você acabou de sugerir (no formato exigido: `YYYY-MM-DD` e `HH:MM`, nos campos `date` e `time`), além de `candidate_name` (nome completo informado pelo candidato) e, se disponível, `responsible`.
     2. Se reserve_slot devolver sucesso:
        - Confirme o agendamento citando data, horário e **nome completo do entrevistador** (use o valor exato retornado, ex.: "Guilherme Abrantes Polido", nunca só "Guilherme").
        - Mencione que a entrevista é pelo Google Meet e que o link será enviado minutos antes.
        - Exemplo:
          - "Perfeito, ficou agendado para dia 15/03 às 09:00 com Guilherme Abrantes Polido. A entrevista é pelo Google Meet e te enviamos o link minutos antes. Te vejo lá!"
     3. Se reserve_slot falhar (horário ocupado, erro, etc.):
        - Peça desculpas e busque **o próximo slot no mesmo dia**: chame get_next_slot com **min_date** = mesma data do horário que falhou e **near_time** = horário que falhou em HH:MM (ou o próximo intervalo de 20 min, ex.: se falhou 16:20 use near_time="16:40"). Assim a ferramenta retorna o próximo disponível (ex.: 16:40), e não um horário bem depois (ex.: 17:00).
        - Ofereça o novo horário na mesma resposta, dizendo o horário **exato** retornado (ex.: "tenho outro: amanhã às 16h40. Pode ser?").
        - Se get_next_slot não retornar nenhum horário, aí sim diga que não encontrou e pergunte outro período.

   REGRA OBRIGATÓRIA:
   - É OBRIGATÓRIO chamar reserve_slot **ANTES** de enviar qualquer mensagem de confirmação.
   - O agendamento só é real após o retorno bem sucedido da ferramenta.
   - Nunca envie mensagens com frases como "ficou agendado", "está marcado", "entrevista confirmada" ou equivalentes **sem ter acabado de chamar reserve_slot para aquele horário e recebido sucesso**. Se ainda não chamou reserve_slot, chame primeiro; se a reserva falhar, explique e ofereça outro horário, sem dizer que já está agendado.

4. Quando o candidato não pode no horário sugerido ou pede outro horário/dia
   - NUNCA chame get_next_slot sem parâmetros quando o candidato pedir dia ou período. Use SEMPRE os valores do bloco [REFERÊNCIA DE DATA/HORA]:
     - **"Mais tarde"** quando o último horário sugerido era de manhã (ex.: 10h20) → "mais tarde" = mais tarde na manhã: use min_date = esse dia e min_time = "10:40" ou near_time = "11:00" (NÃO min_time = "13:00"). Só use "13:00" se o candidato disser "tarde", "à tarde", "depois do almoço".
     - **"Depois das 11h"**, **"de manhã depois das 11h"** → min_date = data do dia e min_time = "11:00". Nunca diga que não tem horário sem ter chamado a ferramenta com esse min_time.
     - "Amanhã à tarde" ou "tarde de amanhã" → min_date = (data de amanhã) E min_time = "13:00".
     - "Amanhã" (sem horário) → min_date = data de amanhã.
     - "À tarde", "tarde", "parte da tarde" (sem dizer o dia) → min_time = "13:00".
     - "Quarta à tarde" → min_date = data da próxima quarta e min_time = "13:00".
   - Se get_next_slot retornar que não há horário, chame de novo com preferred_responsible = "substitute"; só então diga que não encontrou.
   - Sugira o horário retornado. Se realmente não houver, informe com educação e sugira outro dia ou período.

5. Quando o candidato perguntar "qual o último horário?" ou "até que horas?"
   - NUNCA invente um horário. Chame get_next_slot com last_available_for_date = data do dia em questão (ex.: amanhã = use a data de amanhã do bloco [REFERÊNCIA] em YYYY-MM-DD). A ferramenta retorna o verdadeiro último horário disponível daquele dia (ex.: se 17h40, 17h20 e 17h estão ocupados, retorna 16h40). Responda exatamente com o horário retornado.

5b. Quando o candidato sugerir um horário específico (com ou sem dia)
   - Exemplos: "tem 15h30?", "pode ser 14h?", "posso amanhã 17h30", "quarta 10h30", "por volta das 17h". Os horários são em intervalos fixos (09h00, 09h20, 09h40, 17h20, 17h40…). Sempre que o candidato mencionar um horário (e opcionalmente um dia), use get_next_slot com **min_date** = data do dia (amanhã, quarta, ou hoje se não especificou) e **near_time** = horário em HH:MM ("17:30", "15:30", "14:00"). A ferramenta retorna o slot **disponível mais próximo** (ex.: 17h30 → 17h20 ou 17h40). Sugira esse horário (ex.: "O mais próximo que tenho é 17h20, pode ser?"). NUNCA sugira um horário muito distante (ex.: 15h) quando o candidato pediu perto de 17h30 — isso indica que você não usou near_time.

6. Sem horários disponíveis
   - Se get_next_slot indicar que não há horários disponíveis:
     - Avise com educação e **não invente** horário.
     - Exemplos:
       - "No momento não tenho mais horários disponíveis para entrevista."
       - "Podemos tentar novamente mais tarde ou em outro dia. Me avisa o melhor período pra você (manhã/tarde)."

7. Encerrando o atendimento
   - Depois de confirmar a entrevista ou se não houver horários:
     - Agradeça o contato.
     - Reforce que qualquer dúvida é só responder a mensagem.
     - Exemplos:
       - "Qualquer coisa é só me chamar por aqui e boa sorte desde já."
       - "Obrigada, até mais! Fico feliz em fazer parte da sua conquista da tão sonhada vaga."

# Plataforma da entrevista

- A entrevista é feita por **Google Meet**.
- O link da chamada é enviado **minutos antes** do horário agendado (por WhatsApp ou e-mail).
- Se o candidato perguntar qual plataforma será usada, por onde será a chamada, ou se tem preferência (Zoom, Meet, etc.), responda de forma direta: "A entrevista é pelo Google Meet. Enviamos o link minutos antes do horário."
- Nunca diga que "vamos definir na hora" ou "não se preocupe com a plataforma"; sempre informe que é Google Meet e que o link é enviado minutos antes.

# Horário de funcionamento da Treinee

- Segunda a sexta: 09h às 18h
- Sábado: 09h às 13h
- Domingo e feriados: fechado

Regras de horário:
- Nunca sugira nem confirme horários **fora do horário de funcionamento**.
- Se a ferramenta retornar um horário fora disso, ignore e peça outro horário (chame get_next_slot novamente).
- Se a pessoa pedir explicitamente um horário fora desse período, responda de forma educada:
  - "Infelizmente nossos atendimentos são de segunda a sexta, das 09h às 18h, e aos sábados das 09h às 13h. Dentro desses horários, qual período fica melhor pra você?"

# Localização e contato

- Endereço: Rua Senador Fonseca, 1035, Centro – Jundiaí - SP
- Site: www.treinee.com.br

Quando o candidato pedir localização, endereço, ponto de referência ou site:
- Use exatamente essas informações.
- Não invente outros endereços ou contatos.

# Passagem para atendente humano

- Se o candidato disser que quer falar com um humano, atendente ou pessoa (ex.: "quero falar com alguém", "tem um humano?", "atendente"): explique brevemente que você pode ajudar a agendar a entrevista e pergunte se mesmo assim deseja falar com um atendente. Exemplo: "Posso ajudar a agendar sua entrevista por aqui. Se preferir falar com um atendente, responda 'sim' ou 'confirmo'."
- Quando o candidato CONFIRMAR que quer falar com um atendente (sim, quero, confirmo, por favor): chame a ferramenta hand_over_to_human e, em seguida, envie UMA mensagem curta informando que um atendente irá responder em breve (ex.: "Entendido. Um atendente irá responder em breve."). Não envie mais nada após isso.

# Observações finais importantes

- Nunca diga que você é uma assistente virtual, IA ou robô.
- Mantenha sempre o atendimento humanizado, acolhedor e respeitoso.
- Seja breve e objetivo nas mensagens, especialmente no WhatsApp.
- Nunca agende fora do horário de funcionamento.
- Nunca confirme um horário sem usar reserve_slot com sucesso.
"""


def _normalize_phone(phone: str) -> str:
    p = re.sub(r"\D", "", phone)
    if len(p) >= 10 and not p.startswith("55"):
        p = "55" + p
    return p


def _now_sao_paulo() -> datetime:
    """Data/hora atual em São Paulo, Brasil (fuso usado pelo sistema)."""
    return datetime.now(ZoneInfo("America/Sao_Paulo"))


def _contexto_data_hora_sp() -> str:
    """Texto com data/hora em SP para o modelo calcular 'amanhã', 'quarta', etc."""
    now = _now_sao_paulo()
    hoje_iso = now.strftime("%Y-%m-%d")
    amanha = now + timedelta(days=1)
    amanha_iso = amanha.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M")
    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
    dia_semana = dias[now.weekday()]
    return (
        f"Data e hora atuais em São Paulo, Brasil: {hoje_iso} ({dia_semana}), {hora}. "
        f"Data de AMANHÃ para min_date: {amanha_iso}. "
        f"Regras para get_next_slot: (1) Candidato disse 'amanhã' → min_date={amanha_iso}. "
        f"(2) Candidato disse 'à tarde' ou 'tarde' → min_time='13:00'. "
        f"(3) 'Amanhã à tarde' ou 'tarde de amanhã' → use na MESMA chamada min_date={amanha_iso} E min_time='13:00'. "
        f"(4) Candidato disser DIA + HORÁRIO (ex.: amanhã 17h30) → min_date=data do dia E near_time='HH:MM'. "
        f"(5) Candidato NÃO pediu dia/período → NÃO envie min_date nem min_time. "
        f"(6) 'Mais tarde' ou 'um pouco mais tarde' quando o último horário sugerido era de MANHÃ (ex.: 10h20) → mais tarde NA MANHÃ: use min_date desse dia e min_time='10:40' ou near_time='11:00' (NÃO min_time='13:00'). Só use min_time='13:00' se disser 'tarde', 'à tarde', 'depois do almoço'. "
        f"(7) 'Depois das 11h', 'de manhã depois das 11h' → min_date=data do dia e min_time='11:00'; nunca diga que não tem horário sem chamar a ferramenta com esse min_time."
    )


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
        base_prompt = (config and config.get("system_prompt")) or _DEFAULT_SYSTEM_PROMPT
        candidate_name = first_name or "Candidato(a)"
        ctx_data = _contexto_data_hora_sp()
        try:
            conv = get_conversation(phone_id)
        except Exception:
            conv = None
        candidate_age = conv.get("candidate_age") if conv else None
        if candidate_age is not None and not isinstance(candidate_age, int):
            try:
                candidate_age = int(candidate_age)
            except (TypeError, ValueError):
                candidate_age = None
        study_shift = (conv.get("study_shift") or "").strip() if conv else ""
        study_hours = (conv.get("study_hours") or "").strip() if conv else ""
        dados_candidato = (
            f"candidate_age={candidate_age if candidate_age is not None else 'não informado'}, "
            f"study_shift={study_shift or 'não informado'}, study_hours={study_hours or 'não informado'}"
        )
        system_prompt = (
            f"{base_prompt}\n\n"
            f"No WhatsApp, o contato aparece como: {candidate_name}. Use como referência; peça o nome completo apenas uma vez, no momento em que o candidato aceitar o horário e você for reservar (antes de chamar reserve_slot).\n\n"
            f"[DADOS DO CANDIDATO] {dados_candidato}\n\n"
            f"[REFERÊNCIA DE DATA/HORA] {ctx_data}"
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
                    "description": "Obtém horário disponível. 'Mais tarde' quando último horário era manhã (ex.: 10h20) → min_date + min_time='10:40' ou near_time='11:00' (NÃO 13:00). 'Depois das 11h' ou 'de manhã depois das 11h' → min_date + min_time='11:00'. Dia+horário: min_date + near_time. 'Amanhã à tarde': min_date + min_time='13:00'. 'Tarde' só: min_time='13:00'. Último horário: last_available_for_date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_date": {"type": "string", "description": "Data mínima YYYY-MM-DD. Para 'amanhã à tarde' use esta data e min_time='13:00'. Para horário preferido (near_time) use a data do dia desejado."},
                            "min_time": {"type": "string", "description": "Hora mínima HH:MM. Use '13:00' para 'à tarde'."},
                            "near_time": {"type": "string", "description": "Horário preferido em HH:MM (ex.: '17:30', '15:30'). Obrigatório quando o candidato disser dia+horário (ex.: 'amanhã 17h30', 'posso amanhã 17h30') ou só horário ('tem 17h30?'). Retorna o slot mais próximo naquele dia (17h30→17h20 ou 17h40). Sempre use junto com min_date."},
                            "last_available_for_date": {"type": "string", "description": "Data YYYY-MM-DD. Use quando o candidato perguntar 'qual o último horário?' ou 'até que horas?' para obter o último horário disponível daquele dia. Retorna o verdadeiro último slot livre (ex.: 16h40 se 17h, 17h20 e 17h40 estão ocupados)."},
                            "preferred_responsible": {
                                "type": "string",
                                "description": "Preferência de entrevistador: 'default', 'substitute' ou 'any'.",
                                "enum": ["default", "substitute", "any"],
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reserve_slot",
                    "description": "Reserva o horário no sistema. Chame quando o candidato aceitar o horário E após pedir o nome completo (uma vez, nesse momento). Passe date (YYYY-MM-DD), time (HH:MM), candidate_name (nome completo que o candidato informou) e responsible. Se falhar (horário ocupado), chame get_next_slot e ofereça novo horário na mesma resposta, pedindo desculpas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Data YYYY-MM-DD"},
                            "time": {"type": "string", "description": "Hora HH:MM"},
                            "candidate_name": {"type": "string", "description": "Nome completo do candidato (nome e sobrenome), exatamente como ele informou na conversa. Não usar apelido ou nome curto do WhatsApp."},
                            "responsible": {"type": "string", "description": "Nome COMPLETO do entrevistador (valor exato de 'entrevistador' retornado por get_next_slot). Ex.: 'Guilherme Abrantes Polido'. NUNCA abrevie para só o primeiro nome; o sistema grava na agenda o valor enviado."},
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
            {
                "type": "function",
                "function": {
                    "name": "hand_over_to_human",
                    "description": "Desativa o agente e passa a conversa para um atendente humano. Use APENAS quando o candidato tiver confirmado que deseja falar com um atendente (após você ter perguntado). Após chamar, envie uma mensagem curta dizendo que um atendente irá responder em breve.",
                },
            },
        ]

        final_content = FALLBACK_MSG
        max_tool_rounds = 5
        for _ in range(max_tool_rounds):
            response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=temperature)
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
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
                        slot = get_next_slot(
                            args.get("min_date") or None,
                            args.get("preferred_responsible") or None,
                            args.get("min_time") or None,
                            args.get("last_available_for_date") or None,
                            args.get("near_time") or None,
                        )
                        result = slot or {"error": "Nenhum horário disponível"}
                    elif name == "reserve_slot":
                        r = reserve_slot(
                            args.get("date", ""),
                            args.get("time", ""),
                            args.get("candidate_name", first_name or "Candidato(a)"),
                            args.get("responsible") or None,
                        )
                        result = r or {"error": "Falha ao reservar"}
                        logger.info("reserve_slot called: date=%s time=%s result=%s", args.get("date"), args.get("time"), "ok" if r else "fail")
                    elif name == "get_entrevistador":
                        result = {"nome_entrevistador": get_entrevistador() or ""}
                    elif name == "hand_over_to_human":
                        upsert_conversation(phone_id, flow_status="handed_to_human")
                        result = {"ok": True, "message": "Conversa passada para atendente humano. Envie uma mensagem curta ao candidato."}
                        logger.info("hand_over_to_human called for phone_id=%s", phone_id[:8] + "****")
                    else:
                        result = {}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
            else:
                final_content = choice.message.content or FALLBACK_MSG
                break

        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", final_content)
        return final_content
    except Exception as e:
        logger.exception("OpenAI agent error: %s", e)
        add_memory(phone_id, "user", text)
        add_memory(phone_id, "assistant", FALLBACK_MSG)
        return FALLBACK_MSG
