# Vitaliza — App de Inteligência de Retenção

Aplicativo web preditivo-generativo que classifica risco de churn de assinantes da Vitaliza
e gera explicação + prescrição em linguagem natural via Google Gemini.

**Entrega da Semana 10 · Trilha de Tecnologia · MBA Inteli · Módulo 2 · Grupo 3**

---

## O que ele faz

Recebe os dados comportamentais de um assinante (frequência de uso, tipo de contrato, lifetime, etc.)
e devolve em uma única tela:

1. **Score 0–100** de probabilidade de churn (Random Forest, AUC 0,945).
2. **Classificação em faixa de risco** (Crítico / Alto / Médio / Baixo).
3. **Top 5 drivers SHAP** que contribuíram para a predição, com direção e magnitude.
4. **Briefing gerado por LLM (Gemini)** em português: diagnóstico, drivers explicados,
   ação prescrita (com oferta específica do segmento) e cuidados LGPD/Sleeping Dog.

Tem dois modos:

- **🔍 Análise individual** — formulário para um assinante (ou perfil pré-definido), pensado para
  uso pelos times de CS, growth e produto durante o trabalho.
- **📊 Análise em lote** — upload de CSV ou uso do dataset de 4.000 usuários do case;
  gera distribuição por faixa, histograma de scores, tabela de calibração e export CSV.

---

## Setup

### 1. Pré-requisitos
- Python 3.10+
- Pip ou conda

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar a chave do Gemini (opcional, mas recomendado)

Obter uma API key gratuita em <https://aistudio.google.com>.

```bash
export GOOGLE_AI_API_KEY="sua-chave-aqui"
```

Para Windows (PowerShell):
```powershell
$env:GOOGLE_AI_API_KEY = "sua-chave-aqui"
```

> **Sem a chave**, o app funciona em **modo offline** — devolve respostas template baseadas
> na faixa de risco. Útil para demos e desenvolvimento. A diferença é que a análise não é
> personalizada pelo Gemini real (mantém o mesmo formato, mas o texto é fixo por faixa).

### 4. Rodar o app
```bash
streamlit run app.py
```

A interface abre automaticamente no navegador padrão em `http://localhost:8501`.

---

## Arquitetura

```
vitaliza_app/
├── app.py                  # UI Streamlit (entry point)
├── churn_predictor.py      # Wrapper do modelo + SHAP por usuário
├── gemini_client.py        # Cliente Gemini com fallback offline
├── prompts.py              # Templates de prompt (C.R.E.A.T.E.) + respostas mock
├── modelo_churn.pkl        # Random Forest treinado (Semana 10)
├── scaler.pkl              # MinMaxScaler (não usado pelo vencedor, incluído por compat.)
├── metadata_modelo.json    # Features na ordem correta, métricas, flags
├── dataset_com_scores.csv  # 4.000 usuários do case com scores aplicados
└── requirements.txt
```

### Fluxo de dados

```
┌──────────────────┐
│  Form / CSV      │
│  (13 features)   │
└────────┬─────────┘
         ▼
┌──────────────────────────────┐
│ ChurnPredictor                │
│  1. feature engineering       │   ← adiciona Delta_freq, is_sleeping_dog, freq_ratio
│  2. predict_proba             │
│  3. TreeExplainer SHAP        │
└────────┬──────────────────────┘
         ▼
┌──────────────────┐    ┌────────────────────┐
│ score + drivers  │ ─▶ │ GeminiClient        │
│                  │    │  build_prompt()     │   ← C.R.E.A.T.E.
│                  │    │  → Gemini API       │
│                  │    │  ou resposta_mockada│
└────────┬─────────┘    └────────┬───────────┘
         ▼                       ▼
┌──────────────────────────────────────────┐
│ UI Streamlit:                             │
│  - KPIs (score, faixa, Δfreq, sleeping)  │
│  - Gauge Plotly                           │
│  - Top 5 SHAP em barras                   │
│  - Análise em Markdown                    │
└──────────────────────────────────────────┘
```

---

## Prompt do LLM (framework C.R.E.A.T.E.)

O prompt construído em `prompts.py` segue o framework C.R.E.A.T.E.:

| Componente   | Implementação                                                                |
|--------------|------------------------------------------------------------------------------|
| **C**ontext  | Perfil do assinante (13 features + 3 derivadas) + top 5 SHAP values         |
| **R**ole     | System prompt: analista sênior de retenção e CS da Vitaliza                 |
| **E**xamples | Few-shot único de output esperado (formato e profundidade)                  |
| **A**daptation | pt-BR formal-objetivo, jargão de SaaS de consumo, sem floreio             |
| **T**ype     | Estrutura fixa: DIAGNÓSTICO / DRIVERS PRINCIPAIS / AÇÃO PRESCRITA / CUIDADOS|
| **E**xtras   | 5 princípios não-negociáveis (LGPD, Sleeping Dog, calibração de oferta, ...)|

O system prompt inclui as **regras de calibração de oferta por segmento** descritas no
documento estratégico (Plano de Intervenção de Risco — Trilha de Negócios da Semana 10),
garantindo que a recomendação do LLM seja coerente com a estratégia aprovada.

### Caso especial: Sleeping Dog

Quando `is_sleeping_dog == 1`, o LLM recebe uma instrução explícita no system prompt:

> *"Se o usuário for um Sleeping Dog (paga há > 6 meses sem usar), JAMAIS recomende campanha
> de reativação. A prescrição correta nesse caso é 'não intervir' com explicação da razão."*

Isso evita o erro clássico de marketing de assinatura conhecido como "don't wake the sleeping dogs".

---

## Modelo subjacente

| Item                    | Valor                                                |
|-------------------------|------------------------------------------------------|
| Algoritmo               | Random Forest (200 estimadores, max_depth=10)        |
| AUC-ROC (teste)         | 0,945                                                |
| AUC-ROC (CV 5 folds)    | 0,941 ± 0,010                                        |
| Precision / Recall / F1 | 0,814 / 0,810 / 0,812                                |
| Top 3 features SHAP     | freq_ratio (0,256) · Delta_freq (0,214) · Contract_period (0,094) |

Detalhes completos do treinamento estão no notebook da Entrega 1
(`01_Modelo_Preditivo_Churn.ipynb`).

---

## Calibração do modelo (validada em produção)

Quando rodado no dataset completo de 4.000 usuários (modo lote), o modelo apresenta calibração
quase perfeita por faixa:

| Faixa   | N usuários | Churn predito | Churn observado |
|---------|-----------:|--------------:|----------------:|
| Baixo   |      2.414 |          0,6% |            0,6% |
| Médio   |        323 |          5,3% |            5,3% |
| Alto    |        366 |         53,8% |           53,8% |
| Crítico |        897 |         94,9% |           94,9% |

Significa que, dos 897 usuários classificados como Crítico, 94,9% efetivamente cancelaram —
o modelo identifica corretamente o risco no extremo, que é onde a intervenção mais importa.

---

## Para a apresentação

Sugestão de fluxo de demo em ~5 minutos:

1. **Tela inicial** → mostrar status do modelo (AUC 0,945) e do Gemini na sidebar.
2. **Preset "Lucas — O Evadido"** → analisar → score 97,7% Crítico, drivers SHAP em vermelho,
   análise prescreve win-back imediato com migração semestral 50% off.
3. **Preset "Sleeping Dog"** → analisar → score baixo (5,1%) **mas** flag detectada;
   análise prescreve explicitamente **NÃO INTERVIR** e explica o porquê.
4. **Modo Lote** → mostrar distribuição da base (897 críticos = 22,4%), histograma e
   especialmente a tabela de calibração (predito ≈ observado quase 1:1).
5. **Ver prompt enviado** (botão no modo individual) → mostrar a engenharia de prompt
   em C.R.E.A.T.E.

---

## Limitações conhecidas

- Modelo treinado em snapshot agregado, não em série temporal — features de decaimento de
  frequência por janelas semanais não estão disponíveis até o pipeline unificado da Fase 3
  do roadmap entrar em produção.
- A integração com Gemini usa o modelo `gemini-2.5-flash` (rápido e barato).
  Para análises mais profundas, é possível trocar para `gemini-2.5-pro` ajustando
  `MODELO_PADRAO` em `gemini_client.py`.
- Não há autenticação no app — pressuposto é uso interno em rede privada.
- Não há histórico de análises — cada predição é independente. Para uso em produção,
  recomenda-se persistir as predições em log estruturado para análise posterior.

---

## Próximos passos pós-MBA

- Integrar com o pipeline real (Mixpanel + PostgreSQL + GA4 → BigQuery) descrito na Fase 3.
- A/B test entre as recomendações do LLM vs. ofertas hardcoded — medir lift real.
- Capacidade multi-tenant: mesma arquitetura serve outros produtos B2C de wellness.
- Modelo de propensão a upgrade (não só a churn) — usando o mesmo SHAP pipeline.

---

**MBA Inteli · Módulo 2 · Semana 10 · Grupo 3 · 2026**
