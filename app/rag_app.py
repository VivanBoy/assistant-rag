#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application Gradio pour l'assistant RAG de l'Hôtel de la Promenade.
Répond aux questions des employés à partir de la documentation interne.
"""

import os
import sys
import gradio as gr
import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Ajout du chemin racine pour les imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Configuration des chemins
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "1-rag"

class AssistantRAG:
    """Assistant RAG complet avec retrieval et génération."""
    
    def __init__(self):
        print("Initialisation de l'assistant RAG...")
        self.charger_donnees()
        self.charger_modele_embedding()
        self.charger_index()
        self.charger_llm()
        print("Assistant prêt.")
    
    def charger_donnees(self):
        """Charge les chunks de documentation."""
        self.df_chunks = pd.read_parquet(DATA_PROCESSED / "chunks.parquet")
        print(f"✓ {len(self.df_chunks)} chunks chargés")
    
    def charger_modele_embedding(self):
        """Charge le modèle d'embedding all-MiniLM."""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Modèle d'embedding chargé")
    
    def charger_index(self):
        """Charge l'index FAISS."""
        self.index = faiss.read_index(str(MODELS_DIR / "index.faiss"))
        print(f"✓ Index FAISS chargé ({self.index.ntotal} vecteurs)")
    
    def charger_llm(self):
        """Charge le modèle de génération Llama 3.2 3B."""
        model_path = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé : {model_path}\n"
                "Téléchargez-le avec :\n"
                f"cd {MODELS_DIR} && wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            )
        
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False
        )
        print("Modèle de génération chargé")
    
    def rechercher_contexte(self, question, k=3):
        """Recherche les chunks pertinents pour une question."""
        question_embedding = self.embedding_model.encode([question])
        distances, indices = self.index.search(question_embedding.astype('float32'), k)
        
        contexte = ""
        sources = []
        
        for i, idx in enumerate(indices[0]):
            chunk = self.df_chunks.iloc[idx]
            contexte += f"[Extrait {i+1} - {chunk['source']}]\n{chunk['texte']}\n\n"
            sources.append(chunk['source'])
        
        return contexte, list(set(sources))
    
    def repondre(self, question, max_tokens=500):
        """Pipeline RAG complet."""
        # Retrieval
        contexte, sources = self.rechercher_contexte(question)
        
        # Construction du prompt
        prompt = f"""En utilisant uniquement les extraits de documents fournis ci-dessous, réponds à la question de l'employé.

Extraits de la documentation :
{contexte}

Question de l'employé : {question}

Réponse (base-toi strictement sur les extraits fournis) :"""
        
        # Génération
        output = self.llm(prompt, max_tokens=max_tokens, temperature=0.1)
        reponse = output["choices"][0]["text"].strip()
        
        # Formatage des sources pour l'affichage
        sources_str = "\n".join([f"- {s}" for s in sources])
        
        return reponse, sources_str


# Initialisation de l'assistant (une seule fois au démarrage)
print("Démarrage de l'application...")
assistant = AssistantRAG()

# Interface Gradio
with gr.Blocks(title="Assistant RAG - Hôtel de la Promenade", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Assistant RAG - Hôtel de la Promenade
    
    Posez vos questions sur la documentation interne de l'hôtel :
    - Convention collective
    - Critères d'entretien ménager 4 diamants
    - Politiques internes
    - Procédures de réservation
    - Comité SST
    - Etc.
    
    L'assistant répond uniquement à partir des documents officiels.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Votre question",
                placeholder="Ex: Quels sont les congés de deuil prévus ?",
                lines=2
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=1000,
                    value=500,
                    step=50,
                    label="Longueur max de réponse"
                )
            submit_btn = gr.Button("Poser la question", variant="primary")
        
        with gr.Column(scale=2):
            sources_output = gr.Textbox(
                label="Sources consultées",
                lines=5,
                interactive=False
            )
    
    reponse_output = gr.Textbox(
        label="Réponse",
        lines=10,
        interactive=False
    )
    
    # Exemples de questions
    gr.Markdown("### Exemples de questions")
    examples = [
        ["Quels sont les congés de deuil prévus dans la convention collective ?"],
        ["Quels sont les critères pour un hôtel 4 diamants concernant l'entretien ménager ?"],
        ["Qui sont les membres du comité SST ?"],
        ["Quelle est la procédure de gestion des réservations ?"],
        ["Quels sont les principes de vie internes de l'hôtel ?"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=question_input,
        outputs=[reponse_output, sources_output],
        fn=lambda q: assistant.repondre(q),
        cache_examples=False
    )
    
    # Action au clic
    submit_btn.click(
        fn=assistant.repondre,
        inputs=[question_input, max_tokens],
        outputs=[reponse_output, sources_output]
    )
    
    # Action au focus perdu (optionnel)
    question_input.submit(
        fn=assistant.repondre,
        inputs=[question_input, max_tokens],
        outputs=[reponse_output, sources_output]
    )

# Lancement
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )