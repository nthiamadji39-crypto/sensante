---
title: SenSante
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SenSante
Assistant de pre-diagnostic medical pour le Senegal.

## Description
SenSante utilise le Machine Learning pour aider au
pre-diagnostic des maladies courantes (paludisme,
grippe, typhoide) a partir des symptomes du patient.

## Structure du projet
- `data/` : Donnees patients (CSV)
- `models/` : Modele ML serialise
- `api/` : API FastAPI
- `frontend/` : Interface web
- `notebooks/` : Scripts d'exploration

## Demo en ligne
https://adji39-sensante.hf.space

## Stack
- scikit-learn (modele ML)
- FastAPI (API REST)
- Tailwind CSS (frontend responsive)
- Groq/Llama 3 (explication LLM)
- Docker (conteneurisation)

## Auteur
Adji Nogaye THIAM - L2 GLSI - ESP/UCAD

## Cours
Integration de Modeles IA - Dr. El Hadji Bassirou TOURE