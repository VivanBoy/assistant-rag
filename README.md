# Assistant Intelligent pour l'Hôtel de la Promenade

Un assistant intelligent conçu pour les employés de l'**Hôtel de la Promenade** à [Ottawa](https://www.delapromenade.ca/). Ce projet vise à exploiter les données internes de l'hôtel pour améliorer l'expérience client et faciliter le travail du personnel à travers deux types d'assistants :

- Un **assistant RAG (Retrieval-Augmented Generation)** capable de répondre aux questions factuelles en se basant strictement sur la documentation interne de l'hôtel.
- Un **assistant fine-tuné** entraîné à reproduire le style et le ton spécifiques de l'hôtel à partir de sa FAQ officielle.

Le système repose sur des modèles de langage légers et performants, exécutés localement pour garantir la confidentialité des données. Une interface de démonstration permet d'interagir facilement avec l'assistant RAG.

## Structure du Projet

```
assistant-rag/
├── app/                          # Application de démonstration
│   └── rag_app.py                # Interface Gradio pour l'assistant RAG
├── data/                         # Données du projet
│   ├── part-1/                   # Avis clients (format CSV)
│   ├── part-2/                   # Documentation interne (PDF)
│   ├── part-3/                   # FAQ officielle de l'hôtel (PDF)
│   └── processed/                 # Données prétraitées (chunks, index)
├── models/                       # Modèles et index sauvegardés
│   ├── 1-rag/                     # Index FAISS et métadonnées pour le RAG
│   └── 2-ft/                      # Adaptateurs LoRA pour le fine-tuning
├── notebooks/                     # Notebooks Jupyter
│   ├── 1-analyse-nlp/             # Analyse des avis clients
│   ├── 2-assistant-rag/           # Construction de l'assistant RAG
│   └── 3-assistant-ft/            # Fine-tuning du modèle de style
├── src/                           # Modules Python réutilisables
│   └── utils/                      # Fonctions utilitaires
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## Assistant RAG - Fonctionnement

L'assistant RAG combine un système de recherche vectorielle avec un modèle de génération pour répondre aux questions des employés.

**Étapes clés :**
1.  Les documents PDF (convention collective, politiques, procédures) sont découpés en chunks de 512 caractères.
2.  Ces chunks sont vectorisés à l'aide du modèle `all-MiniLM-L6-v2` et indexés avec FAISS pour une recherche rapide.
3.  Une question posée par l'utilisateur est comparée à l'index pour retrouver les 3 passages les plus pertinents.
4.  Ces passages sont fournis comme contexte au modèle de génération **Llama 3.2 3B** (quantifié en 4-bit), qui formule une réponse naturelle et fidèle aux documents.

**Modèles utilisés pour le RAG :**
- **Embeddings** : `all-MiniLM-L6-v2` (384 dimensions)
- **Génération** : `Llama 3.2 3B` au format GGUF (Q4_K_M), exécuté via `llama-cpp-python`

## Assistant Fine-Tuné - Style et Ton

Un second assistant a été développé pour imiter le style de réponse chaleureux et poétique de l'hôtel, en se basant sur les 30 paires questions-réponses de la FAQ.

**Approche :**
- Fine-tuning du modèle **Qwen2.5 3B** avec la technique **QLoRA** (quantification 4-bit + adaptateurs).
- Entraînement sur 30 exemples avec early stopping pour éviter le surapprentissage.
- Comparaison systématique entre le modèle de base, le modèle fine-tuné et la FAQ originale.

**Résultats :**
- Le fine-tuning a réussi à capturer le ton et le style de l'hôtel (expressions chaleureuses, fluidité).
- Cependant, le modèle fine-tuné invente des informations factuelles (tarifs, horaires) absentes de la FAQ, ce qui le rend inadapté pour une utilisation en production.
- **Conclusion** : L'approche RAG reste préférable pour garantir l'exactitude des réponses, tandis que l'assistant fine-tuné démontre la faisabilité d'adaptation stylistique sur un petit volume de données.

## Installation et Lancement de la Démo

### Prérequis
- Python 3.10 ou supérieur
- 8 Go de RAM minimum (16 Go recommandés)
- (Optionnel) GPU NVIDIA avec 8 Go de VRAM pour accélérer l'inférence
- Git

### Installation
1.  Clonez le dépôt :
    ```bash
    git clone https://github.com/VivanBoy/assistant-rag.git
    cd assistant-rag
    ```

2.  Créez un environnement virtuel (recommandé) :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows : venv\Scripts\activate
    ```

3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

4.  Téléchargez le modèle de génération Llama 3.2 3B au format GGUF :
    ```bash
    # Rendez-vous dans le dossier des modèles
    cd models/1-rag
    wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
    cd ../..
    ```
    *Note : Si `wget` n'est pas disponible, téléchargez manuellement le fichier et placez-le dans `models/1-rag/`.*

### Lancer l'Application de Démonstration (RAG)

L'application Gradio permet d'interagir avec l'assistant RAG via une interface web simple.

```bash
python app/rag_app.py
```

Puis ouvrez votre navigateur à l'adresse : `http://localhost:7860`

Vous pouvez poser des questions comme :
- "Quels sont les congés de deuil prévus dans la convention collective ?"
- "Quels sont les critères pour un hôtel 4 diamants ?"
- "Qui sont les membres du comité SST ?"

## Structure des Données

Les fichiers de données ne sont pas tous inclus dans le dépôt en raison de leur taille. Pour reconstruire le système, assurez-vous que les dossiers suivants contiennent les fichiers nécessaires :

- `data/part-2/` : 6 documents PDF (convention collective, politiques, etc.)
- `data/part-3/` : `Partie 3 - FAQ - Hôtel De la Promenade.pdf`
- `models/1-rag/Llama-3.2-3B-Instruct-Q4_K_M.gguf` : Modèle de génération

Les données prétraitées (chunks, index FAISS) sont disponibles dans `data/processed/`.
