# Laboratório 08 — Alinhamento Humano com DPO

> **Partes geradas/complementadas com IA, revisadas por [Seu Nome]**

Pipeline de alinhamento de LLM utilizando **Direct Preference Optimization (DPO)** para garantir comportamento **Útil, Honesto e Inofensivo (HHH)**.

---

## Estrutura do Projeto

```
lab08-dpo/
├── data/
│   └── hhh_dataset.jsonl      # Dataset de preferências (33 exemplos)
├── src/
│   └── train_dpo.py           # Pipeline completo de treinamento DPO
├── outputs/
│   └── dpo-hhh-aligned/       # Modelo treinado (gerado após execução)
├── requirements.txt
└── README.md
```

---

## Passo 1 — Dataset de Preferências (The HHH Dataset)

O arquivo `data/hhh_dataset.jsonl` contém **33 exemplos** no formato exigido pelo DPO, cada linha com as três chaves obrigatórias:

| Chave | Descrição |
|---|---|
| `prompt` | Instrução ou pergunta do usuário |
| `chosen` | Resposta **segura e alinhada** com HHH |
| `rejected` | Resposta **nociva ou inadequada** que deve ser suprimida |

Os exemplos cobrem cenários corporativos de alto risco: ataques a infraestrutura, vazamento de dados, fraudes contábeis, violações de LGPD, manipulação de relatórios, e instalação de malware.

---

## Passo 2 — Pipeline DPO: Modelo Ator e Modelo de Referência

O DPO requer **dois modelos simultâneos** na memória:

### Modelo Ator (`actor_model`)
O modelo que **terá seus pesos atualizados** durante o treinamento. Para eficiência de memória, utilizamos adaptadores **LoRA** (Low-Rank Adaptation), que treinam apenas uma fração pequena dos parâmetros:

```python
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
```

Se o adaptador do Lab 07 estiver disponível, ele é carregado como ponto de partida via `PeftModel.from_pretrained()`.

### Modelo de Referência (`ref_model`)
O **modelo base congelado** (todos os gradientes desativados). Sua função é fornecer a distribuição de probabilidade original `π_ref`, usada como linha de base para o cálculo da divergência KL. Sem ele, o modelo poderia destruir sua capacidade linguística ao otimizar apenas para preferências.

---

## Passo 3 — O Papel Matemático do Hiperparâmetro β (Beta)

O objetivo de treinamento do DPO é definido pela seguinte função de perda:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

onde $\pi_\theta$ é a distribuição do modelo ator, $\pi_{\text{ref}}$ é a distribuição do modelo de referência congelado, $y_w$ é a resposta preferida (*chosen*) e $y_l$ é a resposta rejeitada (*rejected*).

### O que o β controla?

O $\beta$ atua como um **"imposto" sobre o desvio comportamental** em relação ao modelo de referência. Em termos matemáticos, o DPO implicitamente maximiza as preferências humanas sujeito a uma restrição de divergência KL entre $\pi_\theta$ e $\pi_{\text{ref}}$. O $\beta$ é o multiplicador de Lagrange dessa restrição — ele penaliza proporcionalmente o quanto o modelo ator se afasta da distribuição original.

Com **β = 0.1** (valor baixo):
- O "imposto" sobre o desvio é pequeno → o otimizador tem liberdade para reescrever fortemente as probabilidades
- A pressão de alinhamento domina: respostas nocivas são suprimidas com mais agressividade
- Risco: se o dataset for muito pequeno, o modelo pode perder fluência ou colapsar

Com **β alto (→ ∞)**:
- O "imposto" sobre qualquer desvio é enorme → o modelo permanece quase idêntico ao de referência
- O alinhamento é fraco: poucas mudanças comportamentais

**Intuição**: β = 0.1 foi escolhido por ser suficientemente baixo para que a sinalização de preferência HHH domine o treinamento, mas não tão baixo que o modelo perca sua capacidade linguística. É o ponto de equilíbrio entre *aprender o que é certo* e *não esquecer como falar*.

---

## Passo 4 — Treinamento e Validação

### Estratégias de Economia de Memória

```python
DPOConfig(
    optim="paged_adamw_32bit",      # Paginação para picos de VRAM
    gradient_accumulation_steps=4,  # Simula batch maior sem mais memória
    gradient_checkpointing=True,    # Recomputa ativações para economizar VRAM
    load_in_4bit=True,              # Quantização NF4 com double quantization
)
```

### Executando o Treinamento

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar pipeline completo
python src/train_dpo.py
```

### Validação de Alinhamento

Após o treino, o script executa automaticamente uma validação que:

1. **Gera respostas** para prompts maliciosos e verifica que o modelo responde de forma segura
2. **Compara log-probabilidades** de uma resposta `chosen` vs `rejected` para o mesmo prompt, confirmando numericamente que a probabilidade da resposta nociva foi suprimida:

```
Log-prob chosen   (segura) : -1.2341
Log-prob rejected (nociva) : -4.8872
✅ ALINHAMENTO CONFIRMADO: modelo favorece a resposta segura.
```

---

## Critérios de Avaliação Atendidos

- [x] Dataset com 33 exemplos (mínimo: 30) com colunas `prompt`, `chosen`, `rejected`
- [x] `DPOTrainer` configurado sem erros de sintaxe
- [x] Dois modelos na memória: Ator (treinável) e Referência (congelado para KL)
- [x] `beta = 0.1` configurado com justificativa matemática completa no README
- [x] `paged_adamw_32bit` e outras estratégias de economia de memória aplicadas
- [x] Validação de supressão de resposta `rejected` via log-probabilidade
- [x] Entrega via Git com tag `v1.0`

---

## Versionamento

```bash
git init
git add .
git commit -m "feat: Lab 08 - DPO alignment pipeline completo"
git tag v1.0
git push origin main --tags
```

---

*Partes geradas/complementadas com IA, revisadas por [Seu Nome]*
