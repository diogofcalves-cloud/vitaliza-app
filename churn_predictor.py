"""
churn_predictor.py
Wrapper do modelo Random Forest treinado na Semana 10.
Encapsula carregamento, predição e cálculo de SHAP values por usuário.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')

# Mapeamento amigável dos nomes de features para português (UI e prompt LLM)
FEATURE_LABELS = {
    'gender': 'Gênero',
    'Near_Location': 'Mora perto da unidade',
    'Partner': 'Convênio corporativo',
    'Promo_friends': 'Entrou por indicação',
    'Phone': 'Telefone cadastrado',
    'Contract_period': 'Duração do contrato (meses)',
    'Group_visits': 'Participa de aulas em grupo',
    'Age': 'Idade',
    'Avg_additional_charges_total': 'Gasto adicional médio (R$)',
    'Month_to_end_contract': 'Meses até fim do contrato',
    'Lifetime': 'Tempo de assinatura (meses)',
    'Avg_class_frequency_total': 'Frequência média histórica (aulas/sem)',
    'Avg_class_frequency_current_month': 'Frequência no mês atual (aulas/sem)',
    'Delta_freq': 'Queda de frequência (histórica − atual)',
    'is_sleeping_dog': 'Sleeping Dog (paga sem usar)',
    'freq_ratio': 'Razão de uso (atual/histórica)',
}


def classificar_faixa(score: float) -> str:
    """Classifica score em faixa de risco operacional."""
    if score >= 0.70:
        return 'Crítico'
    if score >= 0.40:
        return 'Alto'
    if score >= 0.20:
        return 'Médio'
    return 'Baixo'


def cor_faixa(faixa: str) -> str:
    """Cor da faixa para uso na UI."""
    cores = {'Crítico': '#C00000', 'Alto': '#ED7D31', 'Médio': '#FFC000', 'Baixo': '#548235'}
    return cores.get(faixa, '#808080')


class ChurnPredictor:
    """
    Wrapper que isola o uso do modelo do resto do app.
    Uma instância carrega o modelo uma vez e atende múltiplas predições.
    """

    def __init__(self, modelo_path='modelo_churn.pkl',
                 scaler_path='scaler.pkl',
                 metadata_path='metadata_modelo.json'):
        base = Path(__file__).parent

        with open(base / modelo_path, 'rb') as f:
            self.modelo = pickle.load(f)
        with open(base / scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(base / metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.features = self.metadata['features']
        self.usa_scaler = self.metadata['usa_scaler']
        self.modelo_nome = self.metadata['modelo_vencedor']

        # Inicializa o explainer SHAP uma vez (custoso)
        self._explainer = shap.TreeExplainer(self.modelo)

    def features_engenheiradas(self, raw_input: dict) -> dict:
        """
        Recebe inputs do form (13 features base) e calcula as 3 derivadas.
        Idêntico ao feature engineering do notebook da Semana 10.
        """
        out = dict(raw_input)
        ftot = out['Avg_class_frequency_total']
        fcur = out['Avg_class_frequency_current_month']

        out['Delta_freq'] = round(ftot - fcur, 3)
        out['is_sleeping_dog'] = int(
            out['Lifetime'] > 6 and fcur < 0.5
        )
        out['freq_ratio'] = round(fcur / (ftot + 0.01), 4)
        return out

    def predict(self, raw_input: dict) -> dict:
        """
        Pipeline completo para um usuário:
        1. Engenharia das 3 features derivadas
        2. Score do modelo
        3. SHAP values
        4. Top features contribuidoras

        Devolve dict pronto para consumo pela UI e pelo prompt LLM.
        """
        feats = self.features_engenheiradas(raw_input)

        # Garantir ordem correta
        x = pd.DataFrame([[feats[f] for f in self.features]], columns=self.features)

        # Score
        if self.usa_scaler:
            x_in = self.scaler.transform(x)
            score = float(self.modelo.predict_proba(x_in)[0, 1])
        else:
            score = float(self.modelo.predict_proba(x)[0, 1])

        # SHAP values (devolve para classe positiva = churn)
        shap_raw = self._explainer.shap_values(x)
        if isinstance(shap_raw, list):
            shap_vals = shap_raw[1][0]
        elif shap_raw.ndim == 3:
            shap_vals = shap_raw[0, :, 1]
        else:
            shap_vals = shap_raw[0]

        # Construir lista de drivers (feature, valor, contribuição SHAP)
        drivers = []
        for f, v_shap in zip(self.features, shap_vals):
            drivers.append({
                'feature': f,
                'label': FEATURE_LABELS.get(f, f),
                'valor': feats[f],
                'shap': float(v_shap),
                'direcao': 'aumenta risco' if v_shap > 0 else 'reduz risco',
            })

        # Ordenar por magnitude absoluta do SHAP
        drivers.sort(key=lambda d: abs(d['shap']), reverse=True)

        faixa = classificar_faixa(score)

        return {
            'score': score,
            'score_pct': round(score * 100, 1),
            'faixa': faixa,
            'cor': cor_faixa(faixa),
            'features_completas': feats,
            'drivers': drivers,
            'top5_drivers': drivers[:5],
            'is_sleeping_dog': bool(feats['is_sleeping_dog']),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score em lote, sem SHAP (mais rápido)."""
        # Engenharia em vetor
        df = df.copy()
        if 'Delta_freq' not in df.columns:
            df['Delta_freq'] = df['Avg_class_frequency_total'] - df['Avg_class_frequency_current_month']
        if 'is_sleeping_dog' not in df.columns:
            df['is_sleeping_dog'] = (
                (df['Lifetime'] > 6) & (df['Avg_class_frequency_current_month'] < 0.5)
            ).astype(int)
        if 'freq_ratio' not in df.columns:
            df['freq_ratio'] = df['Avg_class_frequency_current_month'] / (df['Avg_class_frequency_total'] + 0.01)

        x = df[self.features]
        if self.usa_scaler:
            x_in = self.scaler.transform(x)
            scores = self.modelo.predict_proba(x_in)[:, 1]
        else:
            scores = self.modelo.predict_proba(x)[:, 1]

        df['risk_score'] = scores
        df['risk_score_pct'] = (scores * 100).round(1)
        df['faixa_risco'] = df['risk_score'].apply(classificar_faixa)
        return df


# ============================================================
# Teste isolado quando rodado direto
# ============================================================
if __name__ == '__main__':
    p = ChurnPredictor()
    print(f"Modelo: {p.modelo_nome}")
    print(f"Features ({len(p.features)}): {p.features}")
    print(f"Usa scaler: {p.usa_scaler}\n")

    # Caso 1: usuário de alto risco (Evadido típico)
    evadido = {
        'gender': 0, 'Near_Location': 1, 'Partner': 0, 'Promo_friends': 0,
        'Phone': 1, 'Contract_period': 1, 'Group_visits': 0, 'Age': 27,
        'Avg_additional_charges_total': 80.0, 'Month_to_end_contract': 0.5,
        'Lifetime': 1.5, 'Avg_class_frequency_total': 1.8,
        'Avg_class_frequency_current_month': 0.3,
    }
    r = p.predict(evadido)
    print(f"=== EVADIDO (Lucas) ===")
    print(f"Score: {r['score_pct']}% | Faixa: {r['faixa']}")
    print(f"Sleeping Dog: {r['is_sleeping_dog']}")
    print(f"Top 3 drivers:")
    for d in r['top5_drivers'][:3]:
        sinal = '+' if d['shap'] > 0 else ''
        print(f"  {d['label']:50s} = {d['valor']:>8} | SHAP {sinal}{d['shap']:.4f} ({d['direcao']})")

    # Caso 2: usuário fiel (Carlos típico)
    fiel = {
        'gender': 1, 'Near_Location': 1, 'Partner': 1, 'Promo_friends': 1,
        'Phone': 1, 'Contract_period': 12, 'Group_visits': 1, 'Age': 32,
        'Avg_additional_charges_total': 180.0, 'Month_to_end_contract': 8.0,
        'Lifetime': 12.5, 'Avg_class_frequency_total': 3.5,
        'Avg_class_frequency_current_month': 3.4,
    }
    r = p.predict(fiel)
    print(f"\n=== FIEL (Carlos) ===")
    print(f"Score: {r['score_pct']}% | Faixa: {r['faixa']}")

    # Caso 3: sleeping dog
    sd = {
        'gender': 0, 'Near_Location': 1, 'Partner': 1, 'Promo_friends': 0,
        'Phone': 1, 'Contract_period': 12, 'Group_visits': 0, 'Age': 38,
        'Avg_additional_charges_total': 120.0, 'Month_to_end_contract': 5.0,
        'Lifetime': 14.0, 'Avg_class_frequency_total': 2.0,
        'Avg_class_frequency_current_month': 0.2,
    }
    r = p.predict(sd)
    print(f"\n=== SLEEPING DOG ===")
    print(f"Score: {r['score_pct']}% | Faixa: {r['faixa']}")
    print(f"Sleeping Dog: {r['is_sleeping_dog']}")
