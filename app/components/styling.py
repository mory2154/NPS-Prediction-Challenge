"""
Petits helpers de style réutilisés entre les pages.

Conservent une charte cohérente et centralisent les avertissements métier
(ex : C2 ne doit PAS être utilisé pour cibler les Promoteurs à cause de la
disparité d'équité Phase 11 sur Senior — DI=0.40).
"""

from __future__ import annotations

import streamlit as st

# ============================================================
# Sidebar : sélecteur de modèle (par défaut = C2, bascule vers C1 avec warning)
# ============================================================
def champion_selector(
    label: str = "Modèle",
    default: str = "C2",
) -> str:
    """
    Affiche le sélecteur de champion dans la sidebar.

    Returns
    -------
    "C1" ou "C2"
    """
    choice = st.sidebar.radio(
        label,
        options=["C2 — Production (par défaut)", "C1 — Expérimental (texte+tabulaire)"],
        index=0 if default == "C2" else 1,
        help=(
            "C2 est le modèle recommandé : tabulaire uniquement, pas de risque de fuite, "
            "meilleur rappel Détracteur (0,84). C1 utilise des verbatims synthétiques "
            "générés par LLM — son QWK supérieur est une borne haute, pas une "
            "performance réaliste en production."
        ),
    )
    selected = "C1" if choice.startswith("C1") else "C2"

    if selected == "C1":
        st.sidebar.warning(
            "⚠ **Modèle expérimental actif.** C1 utilise des embeddings de "
            "verbatims synthétiques. Le gain QWK reflète la fidélité du LLM à "
            "la classe cible, pas une valeur prédictive réelle. Également : "
            "C1 montre une disparité d'équité sur Married × Détracteur "
            "(DI=0,72) — voir Équité du modèle."
        )
    return selected


# ============================================================
# Bandeaux contextuels
# ============================================================
def promoter_use_warning():
    """À afficher quand l'utilisateur essaie d'utiliser C2 pour cibler les Promoteurs."""
    st.error(
        "🚨 **Attention : C2 présente une forte disparité d'équité sur le rappel Promoteur**.\n\n"
        "Audit Phase 11 : *Disparate Impact* = 0,40 sur Senior × Promoteur "
        "(rappel 0,23 pour les seniors vs 0,58 pour les non-seniors). "
        "Utiliser C2 pour cibler les promoteurs sous-couvrira systématiquement "
        "le segment senior.\n\n"
        "Décisions appropriées selon l'usage :\n"
        "- ✓ Ciblage Détracteur (rétention) — équitable sur les 3 segments\n"
        "- ⚠ Ciblage Promoteur (parrainage, advocacy) — biaisé sur Senior + Married\n\n"
        "Voir **Équité du modèle** pour le détail complet."
    )


def calibration_note():
    """Note de bas de page rappelant que les probas C2 ne sont pas parfaitement
    calibrées sur Passif/Promoteur, donc le *rang* est plus fiable que la
    proba absolue."""
    st.caption(
        "ℹ Les probabilités de C2 sont bien classées mais pas parfaitement "
        "calibrées sur Passif/Promoteur (Phase 9). Pour des décisions "
        "opérationnelles, préférer les **rangs** aux probas absolues."
    )


def synthetic_verbatim_caveat():
    """Bandeau pour la page À propos — caveat des verbatims."""
    st.warning(
        "**Caveat important — verbatims synthétiques**\n\n"
        "Le jeu de données IBM Telco ne contient pas de verbatims clients. "
        "Pour tester la pipeline texte+tabulaire (C1), la Phase 5 a généré "
        "des verbatims synthétiques avec Qwen2.5-7B-Instruct conditionné sur "
        "les caractéristiques du client ET son score de satisfaction. "
        "Par conséquent, les embeddings de C1 *encodent une partie de la cible "
        "par construction*. La Phase 10 l'a confirmé empiriquement : la "
        "composante PC01 à elle seule contribue 3,9× plus que la feature n°2 "
        "pour la prédiction Promoteur.\n\n"
        "→ Le gain QWK de C1 (+0,30 vs C2) doit être lu comme une "
        "**borne supérieure de performance**, pas comme une estimation de la "
        "valeur réelle avec des verbatims clients authentiques. C'est pour "
        "cela que C2 est recommandé en production."
    )


# ============================================================
# Badge équité : vert ✓ si fair, rouge ⚠ si disparité
# ============================================================
def fairness_badge(
    di: float, eod: float,
    fair_di_lo: float = 0.8, fair_di_hi: float = 1.25,
    fair_eod_max: float = 0.10,
) -> str:
    """Chaîne markdown avec emoji + verdict, utilisable inline."""
    if di != di or eod != eod:  # NaN check
        return "❓ N/A"
    di_ok = fair_di_lo <= di <= fair_di_hi
    eod_ok = abs(eod) < fair_eod_max
    if di_ok and eod_ok:
        return f"✓ Équitable (DI={di:.2f}, EOD={eod:+.2f})"
    return f"⚠ Disparité (DI={di:.2f}, EOD={eod:+.2f})"


# ============================================================
# Note seuil pour le ciblage cohorte
# ============================================================
def top_k_note(k: int, total: int):
    pct = k / total * 100 if total else 0
    st.caption(
        f"Affichage des **{k}** premiers clients ({pct:.1f} % de la population silent, "
        f"N={total:,}). Utiliser le curseur 'Top K' pour ajuster."
    )
