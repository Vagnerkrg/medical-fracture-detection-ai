# 🦴 Medical Fracture Detection AI

Sistema de Inteligência Artificial para **detecção e classificação de fraturas ósseas** em imagens médicas, utilizando Visão Computacional e modelos YOLO.

O projeto implementa um pipeline completo de Machine Learning: exploração do dataset, treinamento do modelo, avaliação de desempenho, explainability (XAI) e uma interface de inferência.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black" alt="YOLO11"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"/>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/status-em%20desenvolvimento-yellow?style=flat-square" alt="status"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license"/>
  <img src="https://img.shields.io/github/last-commit/Vagnerkrg/medical-fracture-detection-ai?style=flat-square" alt="last commit"/>
  <img src="https://img.shields.io/github/languages/top/Vagnerkrg/medical-fracture-detection-ai?style=flat-square" alt="top language"/>
  <img src="https://img.shields.io/github/stars/Vagnerkrg/medical-fracture-detection-ai?style=flat-square&logo=github" alt="stars"/>
</p>

---

## 📌 Visão Geral

A solução utiliza um modelo de detecção de objetos baseado em **YOLO11n** para identificar regiões de fraturas ósseas em imagens de raio-X, funcionando como uma ferramenta de apoio à análise médica capaz de detectar diferentes padrões de fraturas e fornecer previsões automatizadas.

## 🎯 Problema

A análise de imagens médicas exige conhecimento especializado e pode ser um processo demorado. Soluções baseadas em Inteligência Artificial podem auxiliar profissionais através da identificação automática de padrões visuais associados a diferentes tipos de fraturas.

## 🎯 Objetivo

Desenvolver uma aplicação de IA capaz de:

- ✅ Detectar fraturas em imagens ósseas
- ✅ Classificar diferentes tipos de fraturas
- ✅ Executar inferência utilizando um modelo treinado
- ✅ Disponibilizar uma interface simples para utilização
- ✅ Manter um pipeline organizado e reproduzível de Machine Learning

---

## 🧩 Solução Proposta

```
Dataset
   ↓
Exploração e análise dos dados
   ↓
Preparação do dataset YOLO
   ↓
Treinamento YOLO11n
   ↓
Avaliação do modelo
   ↓
Explainability (GradCAM)
   ↓
Modelo treinado (best.pt)
   ↓
Inferência
   ↓
Interface Gradio
```

---

## 📂 Dataset

**Origem:** Human Bone Fractures Multi-modal Image Dataset (HBFMID)
Formato preparado para treinamento YOLO.

```
Bone Fractures Detection/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Classes Detectadas

| # | Classe |
|---|---|
| 1 | Comminuted |
| 2 | Greenstick |
| 3 | Healthy |
| 4 | Linear |
| 5 | Oblique Displaced |
| 6 | Oblique |
| 7 | Segmental |
| 8 | Spiral |
| 9 | Transverse Displaced |
| 10 | Transverse |

---

## ⚙️ Pipeline de Machine Learning

### 1. Dataset
Imagens médicas + anotações YOLO.

### 2. Pré-processamento
- Organização das imagens
- Validação das labels
- Configuração YOLO via `data.yaml`

### 3. Treinamento

| Parâmetro | Valor |
|---|---|
| Modelo | YOLO11n |
| Epochs | 50 |
| Image Size | 640 |
| Batch Size | 16 |

```bash
python scripts/train.py
```

### 4. Avaliação

Resultados gerados em `models/runs/train/`, incluindo pesos, métricas, gráficos de treinamento e resultados de validação.

Modelo final: `models/runs/train/weights/best.pt`

### 5. Explainability

Análise de interpretabilidade (Explainable AI) para:
- Visualizar regiões relevantes da imagem
- Entender decisões do modelo
- Aumentar transparência das previsões

### 6. Inferência

```bash
python scripts/predict.py caminho/da/imagem.jpg
```

Exemplo de retorno:

```
Class: Spiral
Confidence: 89.87%
```

---

## 🖥️ Aplicação

Interface desenvolvida com **Gradio**, responsável por receber a imagem, executar a predição e exibir o resultado ao usuário.

```bash
python -m src.app.interface
```

Disponível em: `http://127.0.0.1:7860`

---

## 🗂️ Estrutura do Projeto

```
medical-fracture-detection-ai/
├── data/
├── models/
│   └── runs/
│       └── train/
│           └── weights/
│               └── best.pt
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/
│   └── app/
│       └── interface.py
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## 🚀 Instalação

```bash
git clone https://github.com/Vagnerkrg/medical-fracture-detection-ai.git
cd medical-fracture-detection-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🛠️ Tecnologias Utilizadas

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=flat-square&logo=ultralytics&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/Explainable%20AI-XAI-8A2BE2?style=flat-square"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/>
</p>

---

## 🏗️ Arquitetura

```
                Dataset
                   |
                   v
          Training Pipeline
             train.py
                   |
                   v
             YOLO Model
            best.pt
                   |
                   v
          Prediction Pipeline
            predict.py
                   |
                   v
          Application Layer
            Gradio UI
```

---

## 🗺️ Roadmap Concluído

**M1 - Dataset Analysis**
✅ Exploração do dataset · ✅ Estatísticas · ✅ Visualização das imagens

**M2 - Model Development**
✅ Pipeline YOLO · ✅ Treinamento · ✅ Avaliação · ✅ Explainability

**M3 - Application**
✅ Interface Gradio · ✅ Inferência do modelo

**M4 - Engineering Structure**
✅ Separação treinamento/predição · ✅ Scripts independentes

---

## 📈 Status do Projeto

Projeto em evolução com foco em boas práticas de:

- Machine Learning Engineering
- Computer Vision
- Explainable AI
- Reprodutibilidade
- Organização de pipelines de IA

---

## 👤 Autor

**Vagner Ferreira**
Data Scientist | Data Engineer | AI/LLM Engineer

<p align="left">
  <a href="https://www.linkedin.com/in/vagnerferreiradata"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
  <a href="https://github.com/Vagnerkrg"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/></a>
</p>

Projeto desenvolvido para estudos e aplicação prática em Inteligência Artificial e Visão Computacional.