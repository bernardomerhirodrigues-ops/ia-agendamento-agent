# Agente IA Agendamento (WhatsApp)

Agente em Python (FastAPI) que agenda entrevistas via WhatsApp: recebe mensagens por webhook, consulta horários disponíveis na API do sistema, sugere um horário por vez e, ao aprovar, reserva e envia confirmação.

## Deploy no Render

1. Crie um **Web Service** no [Render](https://render.com).
2. Conecte o repositório (pasta `ia_agendamento_agent` ou raiz do repo).
3. **Build Command:** `pip install -r requirements.txt` (ou deixe em branco se usar raiz e tiver `requirements.txt` na pasta).
4. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Ou use o **Procfile**: Render detecta automaticamente `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. **Root Directory:** se o repositório for a raiz do projeto, defina como `ia_agendamento_agent` (ou a pasta onde estão `app/` e `requirements.txt`).

### Variáveis de ambiente no Render

| Variável | Obrigatório | Descrição |
|----------|-------------|-----------|
| `MYSQL_HOST` | Sim | Host do MySQL (mesmo do sistema PHP). |
| `MYSQL_USER` | Sim | Usuário do banco. |
| `MYSQL_PASSWORD` | Sim | Senha do banco. |
| `MYSQL_DATABASE` | Sim | Nome do banco. |
| `BASE_URL` | Sim | URL base do sistema (ex: `https://seusite.com.br/public`). |
| `API_KEY` | Sim | Mesma chave configurada em Configurações > IA Agendamento. |
| `WEBHOOK_SECRET` | Recomendado | Mesmo valor do campo "Código secreto do webhook" na UI. |
| `OPENAI_API_KEY` | Opcional | Usado se ativar fluxo com OpenAI; pode ser configurado na UI. |

## Endpoints

- **GET /health** – Health check (Render e testes).
- **POST /webhook/whatsapp** – Recebe eventos do WhatsApp; validar com `X-Webhook-Secret` se configurado.
- **POST /webhook/test** – Simula um teste de webhook (útil para o botão "Testar webhook" na UI).

## URL do webhook no provedor WhatsApp

Após o deploy, use no provedor do WhatsApp (Meta, n8n, Evolution, etc.):

```
https://SEU-SERVICO.onrender.com/webhook/whatsapp
```

Na UI do sistema (Configurações > IA Agendamento), preencha a **URL do Render** (ex: `https://SEU-SERVICO.onrender.com`), salve e copie a **URL do webhook** exibida para configurar no provedor.

## Migrações no banco

Execute no banco do sistema (onde roda o PHP) o script:

```
sql/create_ia_agendamento_tables.sql
```

Isso cria as tabelas `ia_agendamento_config`, `ia_agendamento_conversation`, `ia_agendamento_memory` e `ia_agendamento_processed_messages`, além das permissões do módulo.

## Testes

- Teste de health: `curl https://SEU-SERVICO.onrender.com/health`
- Teste de webhook: use o botão "Testar webhook" em Configurações > IA Agendamento (ou `POST /webhook/test` com header `X-Webhook-Secret` se configurado).
