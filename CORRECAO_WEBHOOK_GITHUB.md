# Correção do webhook para aplicar no GitHub

O erro `AttributeError: 'dict' object has no attribute 'strip'` ocorre porque o provedor do WhatsApp envia `text` como objeto `{"body": "mensagem"}`. O código antigo chamava `.strip()` nesse objeto.

## O que fazer no GitHub

1. Abra o repositório no GitHub.
2. Vá até **ia_agendamento_agent/app/main.py**.
3. Clique no ícone de **lápis** (Edit this file).
4. **Localize** o trecho abaixo (por volta da linha 106–110):

```python
    message_id = payload.get("message_id") or ""
    phone = payload.get("phone") or ""
    text = (payload.get("text") or "").strip()
    first_name = (payload.get("first_name") or "").strip()
```

5. **Substitua** esse trecho inteiro por:

```python
    message_id = payload.get("message_id") or ""
    phone = payload.get("phone") or ""
    raw_text = payload.get("text")
    if isinstance(raw_text, dict):
        text = (raw_text.get("body") or raw_text.get("text") or "").strip()
    else:
        text = (str(raw_text or "")).strip()
    first_name = (payload.get("first_name") or "").strip()
```

6. Role até o fim e clique em **Commit changes**.
7. Aguarde o Render fazer o deploy (ou dê **Manual Deploy** no painel do Render).

---

## Variáveis de ambiente necessárias para enviar respostas

O agente envia as respostas via API do seu sistema PHP. No Render, confira:

| Variável   | Descrição                                                                 |
|------------|---------------------------------------------------------------------------|
| `BASE_URL` | URL base do sistema (ex.: `https://seusite.com.br/public`, sem barra no final) |
| `API_KEY`  | Mesma chave definida em **Configurações > IA Agendamento > Chave da API (agente)** |

Sem `BASE_URL` e `API_KEY`, o agente não consegue chamar o PHP e nenhuma resposta é enviada ao WhatsApp.
