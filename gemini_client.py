"""
gemini_client.py
Wrapper para a API do Google AI Studio (Gemini), com fallback automático
para modo offline (resposta mockada) quando a API key não está disponível.

Uso:
    client = GeminiClient()
    if client.online:
        resposta = client.analisar(predicao)
    else:
        resposta = client.analisar_mockado(predicao)
        prompt_usado = client.ultimo_prompt
"""

import os
from prompts import SYSTEM_PROMPT, build_prompt, resposta_mockada

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_DISPONIVEL = True
except ImportError:
    _GENAI_DISPONIVEL = False


class GeminiClient:
    """
    Cliente Gemini com graceful degradation.

    Comportamento:
    - Se GOOGLE_AI_API_KEY (ou GEMINI_API_KEY) estiver setada no ambiente E o SDK estiver
      instalado, o cliente está em modo online e usa o Gemini real.
    - Caso contrário, fica em modo offline e devolve resposta mockada por template
      (definida em prompts.py), permitindo demos e desenvolvimento sem chave.
    """

    MODELO_PADRAO = 'gemini-2.5-flash'

    def __init__(self, api_key: str = None, model_name: str = None):
        self.model_name = model_name or self.MODELO_PADRAO
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.online = False
        self.ultimo_prompt = None
        self.erro = None

        if self.api_key and _GENAI_DISPONIVEL:
            try:
                self._client = genai.Client(api_key=self.api_key)
                self.online = True
            except Exception as e:
                self.erro = f"Falha ao inicializar Gemini: {type(e).__name__}: {e}"
                self.online = False

    def status(self) -> str:
        """Devolve string descritiva para a UI."""
        if self.online:
            return f"🟢 Online ({self.model_name})"
        if not _GENAI_DISPONIVEL:
            return "🔴 SDK google-generativeai não instalado"
        if not self.api_key:
            return "🟡 Modo offline (defina GOOGLE_AI_API_KEY)"
        if self.erro:
            return f"🔴 Erro: {self.erro}"
        return "🔴 Indisponível"

    def analisar(self, predicao: dict) -> str:
        """
        Roteia para a chamada real do Gemini ou para o mock,
        dependendo do estado do cliente. Sempre devolve string em markdown.
        """
        prompt = build_prompt(predicao)
        self.ultimo_prompt = prompt

        if not self.online:
            return resposta_mockada(predicao)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=1500,
                ),
            )
            return response.text
        except Exception as e:
            # Em caso de erro de runtime, cai para o mock e registra
            self.erro = f"{type(e).__name__}: {e}"
            return (
                f"⚠️ Erro na chamada do Gemini ({self.erro}). "
                f"Usando resposta mockada como fallback:\n\n"
                + resposta_mockada(predicao)
            )

    def build_prompt_only(self, predicao: dict) -> str:
        """Útil para debug e para mostrar o prompt na UI."""
        return build_prompt(predicao)


if __name__ == '__main__':
    # Teste rápido
    from churn_predictor import ChurnPredictor

    p = ChurnPredictor()
    evadido = {
        'gender': 0, 'Near_Location': 1, 'Partner': 0, 'Promo_friends': 0,
        'Phone': 1, 'Contract_period': 1, 'Group_visits': 0, 'Age': 27,
        'Avg_additional_charges_total': 80.0, 'Month_to_end_contract': 0.5,
        'Lifetime': 1.5, 'Avg_class_frequency_total': 1.8,
        'Avg_class_frequency_current_month': 0.3,
    }
    pred = p.predict(evadido)

    client = GeminiClient()
    print(f"Status: {client.status()}\n")
    print("=" * 70)
    print("RESPOSTA:")
    print("=" * 70)
    print(client.analisar(pred))
    print("\n" + "=" * 70)
    print(f"Tamanho do prompt enviado: {len(client.ultimo_prompt)} chars")
