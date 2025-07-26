# How to Score and Rank LLMs in a Prediction Market

*(Or: "How does **`ProphetArena`** rank LLMs?")*

## **Author:** Sida Li *(additional authors/reviewers to be added)*

Creating effective benchmarks and arenas to evaluate large language models (LLMs) is often a labor-intensive and meticulous task. Typically, the guiding principle for selecting evaluation metrics has been simplicity and interpretability. For instance, when tasks involve pairwise comparisons—answering the question, "Which LLM is better?"—the Elo rating system provides a clean and intuitive solution. Similarly, for objective benchmarks with clearly verifiable answers, accuracy (average correctness) is straightforward and sufficient.

However, the question of **how to score and rank LLMs based on their probabilistic predictions** introduces more complexity and nuance. Choosing the right metrics becomes a non-trivial yet intriguing challenge. One distinctive strength of our platform, `ProphetArena`, lies precisely in our comprehensive scoring and ranking module. This module implements diverse, principled metrics inspired by statistical theory, utility theory, and psychometrics.

In this post, we'll guide you through the reasoning behind our metric choices and describe how these metrics help us robustly evaluate LLM performance in prediction-market scenarios.

## TL;DR (for readers in a hurry)

* Our default scoring metric in `ProphetArena` is the **Brier score**—a well-established [proper scoring rule](https://en.wikipedia.org/wiki/Scoring_rule). The Brier score captures the core question:

  > *"How well does the predicted probability distribution match reality (the observed outcome)?"*

  It naturally generalizes beyond binary outcomes, assessing both accuracy and calibration.

* We've innovatively introduced a class of **money-earning metrics** as complementary indicators. Intuitively, these metrics simulate the long-term returns of someone consistently betting based purely on the LLM's probability estimates, using the same initial budget per event.

* We incorporate additional metrics such as an **IRT (Item Response Theory) score**, which jointly models each LLM’s predictive ability alongside event-specific difficulty and discrimination parameters, and a **generalized Bradley–Terry model**, a rating system akin to Elo ratings, providing intuitive comparative scores.

* All these metrics are efficiently implemented and packaged into our standalone Python package [`pm_rank`](https://pypi.org/project/pm-rank/), fully documented and open-sourced to facilitate better evaluation of LLMs in general prediction-market environments.

## Scoring Rules: The Grounded Metrics for Probabilistic Predictions

*(See detailed explanation above.)*

## Money Earning: What Practitioners Really Care About

*(See detailed explanation above.)*

## IRT & Bradley–Terry: Statistical Insights into LLM Performance

In addition to the previously mentioned metrics, we also incorporate statistically-grounded methods such as **Item Response Theory (IRT)** and the **Bradley–Terry (BT) model** to gain deeper insights into LLM performance. Unlike simpler metrics, these methods rely on fitting statistical models to data—thus they tend to require larger datasets and careful model fitting procedures.

### Item Response Theory (IRT)

IRT addresses a critical limitation in simpler scoring methods: equal weighting of all prediction events. Using a two-parameter logistic (2-PL) IRT model, we jointly estimate each LLM’s capability parameter alongside the difficulty and discrimination parameters for each prediction event. Higher discrimination parameters indicate events that more effectively distinguish strong from weak predictors, thus implicitly assigning more weight to these informative events.

This approach is highly versatile. The final scoring can be either (1) the directly fitted capability parameters of the LLMs or (2) weighted scoring rules using event-level discrimination parameters. While traditionally the 2-PL IRT model assumes binary outcomes (correct/incorrect), our implementation also accommodates continuous responses, such as directly using the Brier score as the data.

### Generalized Bradley–Terry (BT) Model

The generalized BT model extends traditional pairwise-comparison methods—like those used by LMArena—to our prediction-market setting. Here, each event outcome is viewed as a contest between two "pseudo-teams": a winning team (corresponding to the realized outcome) and a losing team. Each participating LLM contributes fractions of its capability proportional to its predicted probabilities, allocating $p_{ik}$ to the winning team and $1 - p_{ik}$ to the losing team.

We model the winning probability using the BT formulation:

$$
\frac{e^{\theta_w}}{e^{\theta_w} + e^{\theta_l}}
$$

where $\theta_w$ and $\theta_l$ are summed fractional capabilities of the winning and losing teams, respectively. Although this generalization introduces an artificial element by translating a non-pairwise prediction scenario into a pairwise framework, it provides a familiar comparative rating approach. Admittedly, we have not thoroughly explored the statistical properties and convergence guarantees of this generalized BT model, yet it remains a valuable addition to our suite of evaluation tools.
