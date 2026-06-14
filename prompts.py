"""
prompts.py
Templates de prompts para o Gemini, construídos sob o framework C.R.E.A.T.E.
(Context, Role, Examples, Adaptation, Type, Extras).

A função build_prompt() é a única que precisa ser chamada externamente.
"""

# ============================================================
# SYSTEM PROMPT - papel do agente
# ============================================================
SYSTEM_PROMPT = """Você é um analista sênior de retenção e Customer Success da Vitaliza, \
aplicativo brasileiro de assinatura B2C nas verticais de fitness, meditação, nutrição e sono.

Sua tarefa é interpretar a saída de um modelo preditivo de churn (Random Forest com AUC \
de 0,945) e devolver, em português brasileiro, um briefing claro e acionável que combine: \
diagnóstico do perfil, explicação dos drivers de risco, e prescrição de uma intervenção \
específica.

PRINCÍPIOS NÃO-NEGOCIÁVEIS:

1. Especificidade: cite o comportamento real do usuário (ex: "frequência caiu de 3,5 para \
0,3 sessões/sem"), não generalidades.

2. LGPD: nunca cite dados que identifiquem o usuário individualmente. Trabalhe apenas \
sobre features comportamentais.

3. Sleeping Dog: se o usuário for um Sleeping Dog (paga há > 6 meses sem usar), JAMAIS \
recomende campanha de reativação. A prescrição correta nesse caso é "não intervir" \
com explicação da razão.

4. Calibração da oferta: descontos profundos (>=25%) só são apropriados para faixa \
Crítica. Para faixa Alto, prefira upgrade contratual. Para Médio, apenas monitoramento. \
Para Baixo, manutenção.

5. Tom: profissional, objetivo, sem floreio. O leitor é um analista de growth ou um \
gerente de CS — tempo é caro."""


# ============================================================
# CONTEXTO DO MODELO E SEGMENTOS (incluido em todo prompt)
# ============================================================
CONTEXTO_SEGMENTOS = """CONTEXTO DOS SEGMENTOS DE RISCO DA VITALIZA:

| Faixa   | Score      | Tamanho        | Churn esperado | Estratégia                              |
|---------|------------|----------------|----------------|------------------------------------------|
| Crítico | >= 0,70    | ~6.250 (22,4%) | 94,9%          | Win-back imediato + onboarding D0-D30   |
| Alto    | 0,40-0,70  | ~2.567 (9,2%)  | 53,8%          | Intervenção proativa (3 subperfis A/B/C)|
| Médio   | 0,20-0,40  | ~2.260 (8,1%)  | 5,3%           | Monitoramento ativo                     |
| Baixo   | < 0,20     | ~16.852 (60%)  | 0,6%           | Manutenção                              |

OFERTAS POR SUBPERFIL DA FAIXA ALTO:
- Alto.A (mensal + frequência alta): upgrade para semestral (40% off na 1a parcela) ou anual (30% off)
- Alto.B (semestral próximo da renovação): upgrade para anual com 25% off + 1 mês grátis
- Alto.C (queda de engajamento): recomendação de programa via Gemini, sem desconto inicial

OFERTAS PARA FAIXA CRÍTICO:
- Desconto progressivo (30% por 3 meses), pausa 60 dias, downgrade, ou migração semestral 50% off

DRIVERS COMPORTAMENTAIS PRINCIPAIS (top features SHAP do modelo):
1. freq_ratio (razão de uso atual/histórica) — menor = mais risco
2. Delta_freq (queda de frequência) — maior = mais risco
3. Avg_class_frequency_current_month — menor = mais risco
4. Contract_period — 1 mês tem 17,6x mais churn que 12 meses
5. Lifetime — menor = mais risco (early churn)"""


# ============================================================
# EXEMPLO FEW-SHOT
# ============================================================
EXEMPLO_FEWSHOT = """EXEMPLO DE OUTPUT ESPERADO (formato e profundidade):

---
**DIAGNÓSTICO**
Assinante em faixa Crítica (score 89%), perfil "O Evadido": mensal, Lifetime de 1,5 mês, com queda \
acentuada de frequência (de 1,8 para 0,3 sessão/semana — drop de 83%). Sem participação em desafios \
em grupo e sem entrada por indicação. Cobrança vence em 5 dias.

**DRIVERS PRINCIPAIS** (interpretação SHAP)
1. **Queda de frequência (Delta_freq = 1,5)**: contribuição +0,152 — o usuário usava 1,8 vezes/sem \
historicamente e está usando 0,3 agora. É o sinal mais forte de abandono iminente.
2. **Razão de uso (freq_ratio = 0,17)**: contribuição +0,150 — está usando apenas 17% do que usava \
antes. Confirma o sinal anterior em escala relativa.
3. **Contrato mensal (Contract_period = 1)**: contribuição +0,063 — ausência de ancoragem contratual \
amplifica o risco; nada o segura na próxima cobrança.

**AÇÃO PRESCRITA**
Disparar o fluxo de win-back imediato (Caminho A) na próxima visita ao app, antes da cobrança vencer.

- **Oferta**: migração para plano semestral com 50% de desconto na primeira renovação.
- **Canal**: modal in-app no próximo login (mais alta intenção que push ou e-mail neste momento).
- **Mensagem-chave**: "Você experimentou 8 sessões com a gente. Quer continuar — mas com tempo para \
formar o hábito? Plano semestral pela metade do preço só agora."

**CUIDADOS**
- LGPD: a campanha opera sob a base legal de execução de contrato (Art. 7º, V); manter \
transparência sobre uso do score se questionado.
- Se a oferta semestral for recusada, o segundo modal deve oferecer pausa de 60 dias (não desconto \
adicional), preservando margem.
---"""


# ============================================================
# FUNCAO QUE MONTA O PROMPT COMPLETO
# ============================================================
def build_prompt(predicao: dict) -> str:
    """
    Monta o user prompt enviado ao Gemini, com base na saída do ChurnPredictor.
    """
    # Features completas para contexto
    feats = predicao['features_completas']

    # Texto sumarizando o usuário
    perfil = (
        f"- Idade: {feats['Age']} anos\n"
        f"- Tipo de contrato: {feats['Contract_period']} mês(es)\n"
        f"- Tempo de assinatura (Lifetime): {feats['Lifetime']:.1f} meses\n"
        f"- Meses até fim do contrato: {feats['Month_to_end_contract']:.1f}\n"
        f"- Frequência média histórica: {feats['Avg_class_frequency_total']:.2f} aulas/semana\n"
        f"- Frequência no mês atual: {feats['Avg_class_frequency_current_month']:.2f} aulas/semana\n"
        f"- Queda de frequência (Delta_freq): {feats['Delta_freq']:.2f}\n"
        f"- Razão de uso (freq_ratio): {feats['freq_ratio']:.3f}\n"
        f"- Mora perto da unidade: {'Sim' if feats['Near_Location'] else 'Não'}\n"
        f"- Convênio corporativo: {'Sim' if feats['Partner'] else 'Não'}\n"
        f"- Entrou por indicação de amigo: {'Sim' if feats['Promo_friends'] else 'Não'}\n"
        f"- Participa de aulas em grupo: {'Sim' if feats['Group_visits'] else 'Não'}\n"
        f"- Gasto adicional médio: R$ {feats['Avg_additional_charges_total']:.2f}\n"
        f"- Sleeping Dog: {'SIM (paga há > 6m sem usar — atenção)' if feats['is_sleeping_dog'] else 'Não'}"
    )

    # Top 5 drivers em texto
    drivers_txt = ""
    for i, d in enumerate(predicao['top5_drivers'], 1):
        sinal = '+' if d['shap'] > 0 else ''
        drivers_txt += (
            f"{i}. {d['label']} (valor atual: {d['valor']}) | "
            f"contribuição SHAP: {sinal}{d['shap']:.4f} | {d['direcao']}\n"
        )

    # Determinar subperfil (Alto.A/B/C) se aplicável
    subperfil_dica = ""
    if predicao['faixa'] == 'Alto':
        if feats['Contract_period'] == 1 and feats['Avg_class_frequency_current_month'] > 2.0:
            subperfil_dica = "\nIMPORTANTE: este usuário se encaixa no subperfil **Alto.A** (mensal + alta frequência). Oferta indicada: upgrade contratual."
        elif feats['Contract_period'] == 6 and feats['Month_to_end_contract'] <= 1.5:
            subperfil_dica = "\nIMPORTANTE: este usuário se encaixa no subperfil **Alto.B** (semestral próximo de renovação). Oferta indicada: upgrade para anual."
        elif feats['Delta_freq'] > 1.5:
            subperfil_dica = "\nIMPORTANTE: este usuário se encaixa no subperfil **Alto.C** (queda recente de engajamento). Oferta indicada: recomendação de programa via LLM, sem desconto inicial."

    user_prompt = f"""{CONTEXTO_SEGMENTOS}

{EXEMPLO_FEWSHOT}

---

AGORA ANALISE ESTE ASSINANTE:

**Score do modelo**: {predicao['score_pct']}%
**Faixa**: {predicao['faixa']}{subperfil_dica}

**Perfil do assinante**:
{perfil}

**Top 5 drivers (SHAP)**:
{drivers_txt}

Produza o briefing seguindo EXATAMENTE a estrutura do exemplo (DIAGNÓSTICO / DRIVERS PRINCIPAIS / \
AÇÃO PRESCRITA / CUIDADOS). Use markdown. Mantenha cada seção objetiva. Cite números reais do perfil \
acima."""

    return user_prompt


# ============================================================
# RESPOSTA MOCKADA POR FAIXA (modo offline)
# ============================================================
def resposta_mockada(predicao: dict) -> str:
    """
    Resposta template usada quando a API do Gemini não está disponível.
    Permite que o app continue funcional para validação e demos.
    """
    feats = predicao['features_completas']
    drivers = predicao['top5_drivers']
    faixa = predicao['faixa']

    # Caso especial: Sleeping Dog
    if predicao['is_sleeping_dog']:
        return f"""**DIAGNÓSTICO**
Assinante classificado como **Sleeping Dog**: Lifetime de {feats['Lifetime']:.1f} meses e frequência \
atual de apenas {feats['Avg_class_frequency_current_month']:.2f} aulas/sem. Paga consistentemente \
mas não usa o produto. Score baixo ({predicao['score_pct']}%) porque o modelo aprendeu que essa \
população tende a NÃO cancelar — segue por inércia.

**DRIVERS PRINCIPAIS**
- **Flag is_sleeping_dog ativa**: o modelo isola explicitamente esse padrão para evitar campanhas \
contraproducentes.
- **Lifetime alto ({feats['Lifetime']:.1f} meses)**: tempo longo de assinatura reforça a inércia \
de pagamento.
- **Contract_period = {feats['Contract_period']}m**: renovação automática sem necessidade de \
ação consciente do usuário.

**AÇÃO PRESCRITA**
**NÃO INTERVIR.** Qualquer campanha de reativação (push, e-mail "sentimos sua falta", desconto) \
funciona como lembrete de que o usuário paga sem usar — e dispara cancelamento que não ocorreria.

- **Ação única permitida**: e-mail puramente informativo 60 dias antes da renovação anual, com \
curadoria editorial de novos programas (SEM CTA de "volte a usar").
- **Canal**: e-mail.
- **Mensagem-chave**: "Veja o que adicionamos à plataforma desde a última renovação" — sem chamada \
de retorno.

**CUIDADOS**
- Garantir que filtro `is_sleeping_dog == 1` exclua este usuário de TODAS as filas de campanha \
(R3 da matriz de riscos).
- Monitorar a taxa de cancelamento desse segmento na renovação anual (alvo: ≥ 95% de retenção)."""

    # Casos por faixa
    if faixa == 'Crítico':
        return f"""**DIAGNÓSTICO**
Assinante em **faixa Crítica** (score {predicao['score_pct']}%). Perfil compatível com o cluster \
"O Evadido": contrato mensal, queda acentuada de frequência (Delta_freq = {feats['Delta_freq']:.2f}, \
freq_ratio = {feats['freq_ratio']:.2f}), Lifetime de {feats['Lifetime']:.1f} meses. Cancelamento \
provável nos próximos 30 dias sem intervenção.

**DRIVERS PRINCIPAIS**
1. **{drivers[0]['label']}** = {drivers[0]['valor']} | SHAP {drivers[0]['shap']:+.4f} — {drivers[0]['direcao']}.
2. **{drivers[1]['label']}** = {drivers[1]['valor']} | SHAP {drivers[1]['shap']:+.4f} — {drivers[1]['direcao']}.
3. **{drivers[2]['label']}** = {drivers[2]['valor']} | SHAP {drivers[2]['shap']:+.4f} — {drivers[2]['direcao']}.

**AÇÃO PRESCRITA**
Acionar fluxo de **win-back imediato (Caminho A)**.

- **Oferta**: migração para plano semestral com 50% de desconto na primeira renovação \
(preserva margem melhor que desconto sobre o mensal, e aumenta o Lifetime esperado).
- **Canal**: modal in-app no próximo login.
- **Mensagem-chave**: "Vimos que você usa menos do que antes. Quer mais tempo para formar o hábito? \
Semestral pela metade do preço só agora."

**CUIDADOS**
- LGPD: campanha opera sob base legal de execução de contrato. Manter página de transparência sobre \
score (recomendação ANPD).
- Se a oferta semestral for recusada, oferecer pausa de 60 dias como segunda tentativa — não \
desconto adicional."""

    if faixa == 'Alto':
        return f"""**DIAGNÓSTICO**
Assinante em **faixa Alto** (score {predicao['score_pct']}%). Cancelamento ainda não em andamento, \
mas probabilidade elevada. Frequência atual: {feats['Avg_class_frequency_current_month']:.2f} \
aulas/sem (histórica: {feats['Avg_class_frequency_total']:.2f}). Tipo de contrato: \
{feats['Contract_period']} meses.

**DRIVERS PRINCIPAIS**
1. **{drivers[0]['label']}** = {drivers[0]['valor']} | SHAP {drivers[0]['shap']:+.4f}.
2. **{drivers[1]['label']}** = {drivers[1]['valor']} | SHAP {drivers[1]['shap']:+.4f}.
3. **{drivers[2]['label']}** = {drivers[2]['valor']} | SHAP {drivers[2]['shap']:+.4f}.

**AÇÃO PRESCRITA**
Acionar **intervenção proativa (Caminho B)** ajustada ao subperfil.

- **Se contrato mensal e freq atual > 2/sem**: oferta de upgrade para plano semestral com 40% off \
na primeira parcela.
- **Se contrato semestral e ≤ 45 dias para renovação**: oferta de upgrade para anual com 25% off + \
1 mês grátis.
- **Se queda de engajamento (Delta_freq > 1,5)**: recomendação de programa personalizado via LLM, \
sem desconto inicial.
- **Canal**: push notification + e-mail personalizado citando o histórico de uso.

**CUIDADOS**
- Não oferecer desconto agressivo (>= 25%) nesta faixa — canibaliza receita de usuários que \
migrariam para anual sem incentivo."""

    if faixa == 'Médio':
        return f"""**DIAGNÓSTICO**
Assinante em **faixa Médio** (score {predicao['score_pct']}%). Algum sinal de risco, mas ancoragem \
contratual ou comportamental razoável. Risco de cancelamento de 5,3% no semestre observado para \
esta faixa.

**DRIVERS PRINCIPAIS**
1. **{drivers[0]['label']}** = {drivers[0]['valor']} | SHAP {drivers[0]['shap']:+.4f}.
2. **{drivers[1]['label']}** = {drivers[1]['valor']} | SHAP {drivers[1]['shap']:+.4f}.
3. **{drivers[2]['label']}** = {drivers[2]['valor']} | SHAP {drivers[2]['shap']:+.4f}.

**AÇÃO PRESCRITA**
**Monitoramento ativo, sem intervenção tática direta.**

- O score deve ser recalculado diariamente. Se subir para faixa Alto, o usuário é automaticamente \
movido para o fluxo de intervenção proativa.
- **Comunicação genérica**: newsletter semanal segmentada por vertical de interesse (fitness, \
meditação, nutrição ou sono — segmentação que a Vitaliza ainda não implementou plenamente).
- **Canal**: e-mail semanal.

**CUIDADOS**
- ROI individualizado nesta faixa é negativo (custo de personalização > valor preservado). Manter \
investimento concentrado nas faixas Crítico e Alto."""

    # Baixo
    return f"""**DIAGNÓSTICO**
Assinante em **faixa Baixo** (score {predicao['score_pct']}%). Perfil "O Fiel": frequência estável \
({feats['Avg_class_frequency_current_month']:.2f} aulas/sem), Lifetime de {feats['Lifetime']:.1f} \
meses, contrato de {feats['Contract_period']} meses. Risco mínimo (0,6% no semestre para esta \
faixa).

**DRIVERS PRINCIPAIS**
1. **{drivers[0]['label']}** = {drivers[0]['valor']} | SHAP {drivers[0]['shap']:+.4f}.
2. **{drivers[1]['label']}** = {drivers[1]['valor']} | SHAP {drivers[1]['shap']:+.4f}.
3. **{drivers[2]['label']}** = {drivers[2]['valor']} | SHAP {drivers[2]['shap']:+.4f}.

**AÇÃO PRESCRITA**
**Manutenção.** Nenhuma campanha de retenção direta.

- **Oportunidade**: ativar programa de referrals (Promo_friends). Indicação reduz churn do indicado \
de 31,3% para 15,8%.
- **Canal**: e-mail trimestral.
- **Mensagem-chave**: "Indique um amigo e ganhe 1 mês grátis."

**CUIDADOS**
- Custo de comunicação especial é desproporcional ao risco. Investir esforço em programas de \
referência, não em comunicação 1:1."""
