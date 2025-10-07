# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width="40%" height="40%"></a>
</p>

---

# CardioIA – Diagnóstico Automatizado no Estetoscópio Digital

### Fase 2

---

## 👨‍🎓 Integrantes
- [Alexandre Oliveira Mantovani](https://www.linkedin.com/in/alexomantovani)
- [Edmar Ferreira Souza](https://www.linkedin.com/in/)
- [Ricardo Lourenço Coube](https://www.linkedin.com/in/ricardolcoube/)
- [Jose Andre Filho](https://www.linkedin.com/in/joseandrefilho)

## 👩‍🏫 Professores
- Tutor: [Leonardo Ruiz Orabona](https://www.linkedin.com/in/leonardoorabona)
- Coordenador: [André Godoi](https://www.linkedin.com/in/profandregodoi)

---

## 📌 Descrição do Projeto
Esta etapa do projeto **CardioIA** foca na construção de um módulo inteligente que simula o "estetoscópio digital". Combinamos técnicas simples de **NLP**, **classificação de texto** e **governança de dados** para identificar sintomas em descrições curtas de pacientes e propor diagnósticos iniciais. O objetivo é demonstrar como estruturas básicas de dados e algoritmos acessíveis podem apoiar processos de triagem e priorização clínica.

> **Governança & Ética (LGPD)**: todos os dados gerados são fictícios ou anonimizados, destinados exclusivamente ao aprendizado acadêmico. Este material **não** substitui avaliação médica profissional.

---

## 📦 Entregáveis

### 🗣️ Parte 1 — Extração de Sintomas e Diagnóstico Assistido
- **Relatos de pacientes**: `data/relatos_pacientes.txt`
- **Mapa de conhecimento (CSV)**: `data/mapa_sintomas_doencas.csv` com sinônimos e coluna `severity_level` indicando gravidade estimada.
- **Script de inferência**: `src/diagnostico.py`
  - Normaliza texto (remoção de acentos), cruza sintomas com o mapa e gera resumo por relato.
  - Calcula a gravidade mais alta detectada (`baixo`, `moderado`, `alto`, `crítico`).
  - Exporta resultados estruturados (`--export caminho.json|csv`).
  - Exibe estatísticas gerais de cobertura e distribuição de gravidade.

### 📊 Parte 2 — Classificador de Risco com TF-IDF
- **Base rotulada**: `data/classificacao_risco.csv`
- **Notebook**: `notebooks/classificador_risco.ipynb`
  - Pipeline TF-IDF + Regressão Logística (Scikit-learn).
  - Avaliação com acurácia, relatório de classificação e matriz de confusão.
  - Identificação dos principais termos por classe e observações de viés.
- **Script CLI opcional**: `src/classificador_risco_cli.py` para treinar/avaliar via terminal, exportar métricas (`--report`) ou salvar modelo (`--export-model`).

### 🎬 Demonstração em Vídeo
- Link (até 4 minutos, YouTube não listado): `pendente – adicionar após gravação`.

---

## 🧪 Metodologia
1. **Curadoria de Sintomas**: seleção de 10 relatos com diferentes sinais cardíacos e níveis de urgência.
2. **Ontologia Simplificada**: criação de mapa de sinônimos → doenças com etiqueta de gravidade para apoiar triagem.
3. **Normalização Léxica**: remoção de acentos e comparação direta para garantir funcionamento em vocabulário restrito.
4. **Modelagem Supervisonada**: montagem de base balanceada de frases rotuladas (alto vs. baixo risco) e treinamento com Regressão Logística.
5. **Interpretação & Viés**: análise dos termos com maior peso no modelo e registro das limitações da base sintética.

---

## ⚙️ Como Executar

### Parte 1 – Script de Diagnóstico
```bash
pip install -r requirements.txt
python3 src/diagnostico.py
```
- O terminal exibirá sintomas reconhecidos, diagnósticos sugeridos e gravidade máxima por relato.
- Para exportar o resultado estruturado:
  ```bash
  python3 src/diagnostico.py --export saida/diagnosticos.json
  python3 src/diagnostico.py --export saida/diagnosticos.csv --format csv
  ```
- É possível usar arquivos personalizados com `--reports` e `--mapping`.

### Parte 2 – Notebook de Classificação
1. Abra `notebooks/classificador_risco.ipynb` no Jupyter Lab/Notebook.
2. Execute as células na ordem apresentada:
   - Carregamento da base `classificacao_risco.csv` e checagem de balanceamento.
   - Divisão treino/teste e vetorização TF-IDF.
   - Treinamento, métricas quantitativas e inspeção de termos mais relevantes.
   - Predições em novas frases.
3. Ajuste hiperparâmetros ou adicione novos exemplos para investigar variações de desempenho.

### Parte 2 – Execução via CLI (opcional)
```bash
python3 src/classificador_risco_cli.py --report saida/metricas.json --export-model saida/modelo.joblib
```
- Gera métricas no terminal, salva relatório JSON e o pipeline treinado (TF-IDF + Regressão Logística).
- Parâmetros adicionais: `--test-size` e `--random-state`.

---

## 📊 Métricas Observadas
- **Regressão Logística** (baseline): acurácia de validação e métricas de precisão/recall calculadas no *hold-out* interno (ver notebook/CLI).
- **Cobertura de Sintomas**: os 10 relatos possuem pelo menos um sintoma identificado; casos não mapeados geram aviso para revisão manual.
- **Gravidade Sugerida**: distribuição resumida ao final da execução (`crítico`, `alto`, `moderado`, `baixo`).
- **Viés & Limitações**: base sintética com vocabulário limitado → necessidade de expandir dados reais para uso clínico.

---

## 🗂️ Estrutura do Projeto
```
📦 fase2
│
├─ assets/
│   └─ logo-fiap.png
├─ data/
│   ├─ relatos_pacientes.txt
│   ├─ classificacao_risco.csv
│   └─ mapa_sintomas_doencas.csv
├─ notebooks/
│   └─ classificador_risco.ipynb
├─ src/
│   ├─ diagnostico.py
│   └─ classificador_risco_cli.py
├─ requirements.txt
└─ README.md
```

---

## ✅ Requisitos para Execução
- Python **3.10+**
- Dependências listadas em `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 📝 Licença
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">
Este projeto segue o modelo FIAP e está licenciado sob 
<a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer">Attribution 4.0 International (CC BY 4.0)</a>.
</p>
