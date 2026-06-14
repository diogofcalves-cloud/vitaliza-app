"""
app.py
Vitaliza — Inteligência de Retenção
App preditivo-generativo: classifica risco de churn de um assinante e gera
diagnóstico + prescrição em linguagem natural via LLM (Gemini).

MBA Inteli · Módulo 2 · Semana 10 · Grupo 3
"""

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn_predictor import ChurnPredictor, classificar_faixa, cor_faixa, FEATURE_LABELS
from gemini_client import GeminiClient

# ============================================================
# CONFIGURACAO DA PAGINA
# ============================================================
st.set_page_config(
    page_title="Vitaliza · Inteligência de Retenção",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS para refinar visual
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 2.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 500; }
    div.block-container { padding-top: 1.5rem; max-width: 1300px; }
    h1 { color: #1F3864; }
    h2 { color: #2E75B6; }
    h3 { color: #5D7B9D; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHE
# ============================================================
@st.cache_resource
def carregar_predictor():
    return ChurnPredictor()

@st.cache_resource
def carregar_gemini():
    return GeminiClient()

@st.cache_data
def carregar_dataset_demo():
    """Dataset com scores já calculados para o modo lote. Recalcula scores para
    garantir compatibilidade com a nomenclatura atual de faixas."""
    df = pd.read_csv(Path(__file__).parent / 'dataset_com_scores.csv')
    # Recalcular faixa com a função atual (garante nomenclatura limpa)
    if 'risk_score' in df.columns:
        df['risk_score_pct'] = (df['risk_score'] * 100).round(1)
        df['faixa_risco'] = df['risk_score'].apply(classificar_faixa)
    return df


predictor = carregar_predictor()
gemini = carregar_gemini()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Vitaliza")
    st.caption("Inteligência de Retenção · MBA Inteli · G3")
    st.divider()

    st.markdown("**Status do modelo**")
    st.info(f"**{predictor.modelo_nome}** · AUC 0,945  \nFeatures: {len(predictor.features)}")

    st.markdown("**Status do Gemini**")
    st.markdown(gemini.status())

    if not gemini.online:
        with st.expander("🔧 Como ativar Gemini real"):
            st.code(
                "# 1. Obter API key em https://aistudio.google.com\n"
                "# 2. Definir variável de ambiente:\n"
                "export GOOGLE_AI_API_KEY=\"sua-chave-aqui\"\n\n"
                "# 3. Reiniciar o Streamlit",
                language="bash"
            )
            st.caption(
                "Sem a chave, o app opera em modo offline com respostas template "
                "(úteis para demo e desenvolvimento)."
            )

    st.divider()

    st.markdown("**Modo de análise**")
    modo = st.radio(
        "Selecione",
        ["🔍 Análise individual", "📊 Análise em lote"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(
        "Este app é o entregável da trilha de tecnologia da Semana 10. "
        "O modelo subjacente foi treinado em 4.000 assinantes anonimizados; "
        "a interface está pensada para uso pelos times de Growth, CS e Produto."
    )

# ============================================================
# HEADER
# ============================================================
st.title("Inteligência de Retenção")
st.markdown(
    "**Score preditivo de churn + explicação em linguagem natural**, integrado em uma "
    "interface única para os times de Growth, CS e Produto da Vitaliza."
)
st.divider()

# ============================================================
# MODO INDIVIDUAL
# ============================================================
if modo == "🔍 Análise individual":

    col_form, col_resultado = st.columns([1, 1.3], gap="large")

    with col_form:
        st.markdown("### Características do assinante")

        # Presets para facilitar demo
        preset = st.selectbox(
            "Carregar perfil pré-definido (opcional)",
            ["— (preencher manualmente)",
             "Lucas — O Evadido (Crítico)",
             "Carlos — O Fiel (Baixo)",
             "Ana — O Semestral (Alto)",
             "Felipe — O Ativo em Risco (Alto)",
             "Sleeping Dog"]
        )

        # Definir defaults pelo preset escolhido
        defaults = {
            'gender': 0, 'Near_Location': 1, 'Partner': 0, 'Promo_friends': 0,
            'Phone': 1, 'Contract_period': 1, 'Group_visits': 0, 'Age': 29,
            'Avg_additional_charges_total': 100.0, 'Month_to_end_contract': 0.5,
            'Lifetime': 2.0, 'Avg_class_frequency_total': 1.8,
            'Avg_class_frequency_current_month': 1.5,
        }

        if "Lucas" in preset:
            defaults.update({'Contract_period': 1, 'Group_visits': 0, 'Promo_friends': 0,
                             'Age': 27, 'Lifetime': 1.5, 'Month_to_end_contract': 0.5,
                             'Avg_class_frequency_total': 1.8,
                             'Avg_class_frequency_current_month': 0.3,
                             'Avg_additional_charges_total': 80.0})
        elif "Carlos" in preset:
            defaults.update({'Contract_period': 12, 'Group_visits': 1, 'Promo_friends': 1,
                             'Partner': 1, 'Age': 32, 'Lifetime': 12.5,
                             'Month_to_end_contract': 8.0,
                             'Avg_class_frequency_total': 3.5,
                             'Avg_class_frequency_current_month': 3.4,
                             'Avg_additional_charges_total': 180.0})
        elif "Ana" in preset:
            defaults.update({'Contract_period': 6, 'Group_visits': 1, 'Promo_friends': 0,
                             'Partner': 0, 'Age': 30, 'Lifetime': 4.5,
                             'Month_to_end_contract': 1.0,
                             'Avg_class_frequency_total': 2.6,
                             'Avg_class_frequency_current_month': 2.3,
                             'Avg_additional_charges_total': 120.0})
        elif "Felipe" in preset:
            defaults.update({'Contract_period': 1, 'Group_visits': 0, 'Promo_friends': 0,
                             'Age': 26, 'Lifetime': 3.0, 'Month_to_end_contract': 0.5,
                             'Avg_class_frequency_total': 3.2,
                             'Avg_class_frequency_current_month': 3.4,
                             'Avg_additional_charges_total': 60.0})
        elif "Sleeping Dog" in preset:
            defaults.update({'Contract_period': 12, 'Group_visits': 0, 'Promo_friends': 0,
                             'Partner': 1, 'Age': 38, 'Lifetime': 14.0,
                             'Month_to_end_contract': 5.0,
                             'Avg_class_frequency_total': 2.0,
                             'Avg_class_frequency_current_month': 0.2,
                             'Avg_additional_charges_total': 120.0})

        # Form em duas colunas
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Contrato e perfil**")
            contract_period = st.selectbox(
                "Contrato (meses)",
                [1, 6, 12],
                index=[1, 6, 12].index(defaults['Contract_period'])
            )
            lifetime = st.number_input("Lifetime (meses)", 0.0, 30.0,
                                        float(defaults['Lifetime']), step=0.5)
            month_to_end = st.number_input("Meses até fim do contrato", 0.0, 12.0,
                                            float(defaults['Month_to_end_contract']),
                                            step=0.5)
            age = st.slider("Idade", 18, 50, int(defaults['Age']))
            gender = st.selectbox("Gênero",
                                  [(0, "Não informado"), (1, "Informado")],
                                  format_func=lambda x: x[1],
                                  index=defaults['gender'])[0]

        with c2:
            st.markdown("**Engajamento**")
            freq_total = st.number_input(
                "Frequência histórica (aulas/sem)", 0.0, 7.0,
                float(defaults['Avg_class_frequency_total']), step=0.1
            )
            freq_current = st.number_input(
                "Frequência mês atual (aulas/sem)", 0.0, 7.0,
                float(defaults['Avg_class_frequency_current_month']), step=0.1
            )
            add_charges = st.number_input(
                "Gasto adicional médio (R$)", 0.0, 600.0,
                float(defaults['Avg_additional_charges_total']), step=10.0
            )
            group_visits = st.checkbox(
                "Participa de aulas em grupo",
                value=bool(defaults['Group_visits'])
            )
            promo_friends = st.checkbox(
                "Entrou por indicação",
                value=bool(defaults['Promo_friends'])
            )

        st.markdown("**Outros**")
        cn1, cn2, cn3 = st.columns(3)
        with cn1:
            near_location = st.checkbox(
                "Mora perto", value=bool(defaults['Near_Location'])
            )
        with cn2:
            partner = st.checkbox(
                "Convênio corporativo", value=bool(defaults['Partner'])
            )
        with cn3:
            phone = st.checkbox(
                "Telefone cadastrado", value=bool(defaults['Phone'])
            )

        analisar = st.button("🎯 Analisar risco", type="primary",
                             use_container_width=True)

    with col_resultado:
        st.markdown("### Resultado")

        if not analisar:
            st.info(
                "⬅️ Preencha as características à esquerda (ou escolha um perfil "
                "pré-definido) e clique em **Analisar risco** para receber o score, "
                "drivers SHAP e o briefing gerado pelo LLM."
            )
        else:
            # Montar input
            raw = {
                'gender': int(gender),
                'Near_Location': int(near_location),
                'Partner': int(partner),
                'Promo_friends': int(promo_friends),
                'Phone': int(phone),
                'Contract_period': int(contract_period),
                'Group_visits': int(group_visits),
                'Age': int(age),
                'Avg_additional_charges_total': float(add_charges),
                'Month_to_end_contract': float(month_to_end),
                'Lifetime': float(lifetime),
                'Avg_class_frequency_total': float(freq_total),
                'Avg_class_frequency_current_month': float(freq_current),
            }

            with st.spinner("Calculando score e SHAP..."):
                pred = predictor.predict(raw)

            # ============ KPIs ============
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Score de risco", f"{pred['score_pct']}%")
            k2.metric("Faixa", pred['faixa'])
            k3.metric("Δ Frequência", f"{pred['features_completas']['Delta_freq']:.2f}")
            k4.metric("Sleeping Dog", "SIM" if pred['is_sleeping_dog'] else "Não")

            # ============ GAUGE ============
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred['score_pct'],
                number={'suffix': "%", 'font': {'size': 38}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': pred['cor'], 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 20], 'color': '#E2F0D9'},
                        {'range': [20, 40], 'color': '#FFF2CC'},
                        {'range': [40, 70], 'color': '#FCE9D3'},
                        {'range': [70, 100], 'color': '#FBE5E5'},
                    ],
                    'threshold': {
                        'line': {'color': pred['cor'], 'width': 4},
                        'thickness': 0.85,
                        'value': pred['score_pct']
                    }
                },
            ))
            gauge.update_layout(height=240,
                                margin=dict(l=20, r=20, t=20, b=10),
                                paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})

            # ============ TOP DRIVERS (SHAP) ============
            st.markdown("#### Top 5 drivers da predição (SHAP)")
            drivers_df = pd.DataFrame(pred['top5_drivers']).iloc[::-1]
            drivers_df['cor'] = drivers_df['shap'].apply(
                lambda v: '#C00000' if v > 0 else '#548235'
            )

            bar_fig = go.Figure(go.Bar(
                x=drivers_df['shap'],
                y=drivers_df['label'],
                orientation='h',
                marker_color=drivers_df['cor'],
                text=[
                    f"  {'+' if v > 0 else ''}{v:.3f}  ·  valor: {val}"
                    for v, val in zip(drivers_df['shap'], drivers_df['valor'])
                ],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>'
            ))
            bar_fig.update_layout(
                height=280,
                margin=dict(l=10, r=180, t=10, b=10),
                xaxis_title="Contribuição SHAP (+ aumenta risco, − reduz risco)",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            bar_fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#404040')
            st.plotly_chart(bar_fig, use_container_width=True,
                            config={'displayModeBar': False})

            # ============ ANALISE GENERATIVA ============
            st.markdown("---")
            st.markdown("#### 🤖 Análise do agente (Gemini)")

            with st.spinner("Gerando análise..."):
                resposta = gemini.analisar(pred)

            # Container com estilo para a resposta
            st.markdown(resposta)

            # Botões de utilidade
            st.divider()
            cu1, cu2, cu3 = st.columns(3)

            with cu1:
                # Download da análise
                output_md = f"""# Análise de risco · Vitaliza

**Score**: {pred['score_pct']}% · **Faixa**: {pred['faixa']}

## Perfil
{chr(10).join(f"- {FEATURE_LABELS.get(k, k)}: {v}"
              for k, v in pred['features_completas'].items())}

## Análise
{resposta}
"""
                st.download_button(
                    "📄 Baixar análise (Markdown)",
                    output_md,
                    file_name=f"analise_risco_{pred['faixa'].lower()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with cu2:
                # Mostrar prompt enviado
                if st.button("🔍 Ver prompt enviado", use_container_width=True):
                    st.session_state['mostrar_prompt'] = True

            with cu3:
                # Reset
                if st.button("🔄 Nova análise", use_container_width=True):
                    st.rerun()

            if st.session_state.get('mostrar_prompt'):
                with st.expander("Prompt enviado ao Gemini", expanded=True):
                    prompt_atual = gemini.ultimo_prompt or gemini.build_prompt_only(pred)
                    st.code(prompt_atual, language="text")

# ============================================================
# MODO LOTE
# ============================================================
else:
    st.markdown("### Análise em lote")
    st.caption(
        "Faça upload de um CSV com os assinantes ou use o dataset de exemplo (4.000 usuários "
        "com scores já calculados pelo modelo treinado na Semana 10)."
    )

    fonte = st.radio(
        "Fonte dos dados",
        ["Usar dataset de exemplo (4.000 usuários)", "Fazer upload de CSV"],
        horizontal=True
    )

    df = None
    if fonte == "Usar dataset de exemplo (4.000 usuários)":
        df = carregar_dataset_demo()
        st.success(f"Dataset carregado: {len(df):,} usuários".replace(",", "."))
    else:
        up = st.file_uploader("CSV com features do modelo", type=['csv'])
        if up is not None:
            df_raw = pd.read_csv(up)
            with st.spinner("Aplicando modelo..."):
                df = predictor.predict_batch(df_raw)
            st.success(f"{len(df):,} usuários classificados".replace(",", "."))

    if df is not None:
        # KPIs agregados
        dist = df['faixa_risco'].value_counts()
        churn_obs = df.groupby('faixa_risco')['Churn'].mean() if 'Churn' in df.columns else None

        st.markdown("#### Distribuição da base por faixa de risco")
        cols = st.columns(4)
        ordem = ['Baixo', 'Médio', 'Alto', 'Crítico']
        cores = ['#548235', '#FFC000', '#ED7D31', '#C00000']

        for col, faixa, cor in zip(cols, ordem, cores):
            n = int(dist.get(faixa, 0))
            pct = n / len(df) * 100 if len(df) else 0
            with col:
                st.markdown(
                    f"<div style='border-left: 6px solid {cor}; padding: 12px 16px; "
                    f"background: #F8F8F8; border-radius: 4px;'>"
                    f"<div style='color: {cor}; font-size: 13px; font-weight: 600;'>{faixa}</div>"
                    f"<div style='font-size: 28px; font-weight: 700; color: #1F3864;'>{n:,}</div>"
                    f"<div style='color: #606060; font-size: 12px;'>{pct:.1f}% da base</div>"
                    "</div>".replace(",", "."),
                    unsafe_allow_html=True
                )

        st.markdown("&nbsp;")

        # Histograma de scores
        col_h, col_t = st.columns([1.3, 1])
        with col_h:
            st.markdown("#### Distribuição contínua de scores")
            hist = go.Figure(go.Histogram(
                x=df['risk_score'] * 100,
                nbinsx=40,
                marker_color='#2E75B6',
                opacity=0.85
            ))
            # Marcações das faixas
            for x, cor in zip([20, 40, 70], ['#FFC000', '#ED7D31', '#C00000']):
                hist.add_vline(x=x, line_width=2, line_dash='dash', line_color=cor)
            hist.update_layout(
                height=320,
                xaxis_title="Score de risco (%)",
                yaxis_title="Número de assinantes",
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,248,248,0.5)',
            )
            st.plotly_chart(hist, use_container_width=True,
                            config={'displayModeBar': False})

        with col_t:
            st.markdown("#### Top 20 maiores scores")
            top = df.nlargest(20, 'risk_score')[
                ['risk_score_pct', 'faixa_risco', 'Contract_period',
                 'Lifetime', 'Avg_class_frequency_current_month']
            ].rename(columns={
                'risk_score_pct': 'Score %',
                'faixa_risco': 'Faixa',
                'Contract_period': 'Contrato',
                'Lifetime': 'Lifetime',
                'Avg_class_frequency_current_month': 'Freq atual',
            }).reset_index(drop=True)
            st.dataframe(top, use_container_width=True, hide_index=True, height=320)

        # Download
        st.markdown("#### Exportar resultado")
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Baixar CSV com scores",
            csv_buf.getvalue(),
            file_name="vitaliza_scores.csv",
            mime="text/csv"
        )

        # Calibração (se tiver coluna Churn real)
        if churn_obs is not None:
            st.markdown("#### Calibração: churn real observado por faixa predita")
            cal_df = pd.DataFrame({
                'Faixa': ordem,
                'N usuários': [int(dist.get(f, 0)) for f in ordem],
                'Churn real (%)': [
                    round(churn_obs.get(f, 0) * 100, 1) for f in ordem
                ],
                'Esperado pelo modelo (%)': [0.6, 5.3, 53.8, 94.9],
            })
            st.dataframe(cal_df, use_container_width=True, hide_index=True)
            st.caption(
                "Esta tabela mostra que o modelo está bem calibrado: a coluna 'Churn real' "
                "deve estar próxima da 'Esperado pelo modelo'. Divergência grande indica "
                "que a calibração precisa ser revisitada."
            )
