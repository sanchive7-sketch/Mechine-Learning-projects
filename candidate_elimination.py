"""
Candidate-Elimination Algorithm
---------------------------------
Concept learning: finds the version space (set of all hypotheses
consistent with the training examples) by maintaining two boundary sets:

    S = most specific boundary   (most specific hypotheses consistent with data)
    G = most general boundary    (most general hypotheses consistent with data)

Hypothesis representation (conjunction of constraints per attribute):
    '?'  -> any value accepted (most general for that attribute)
    'phi' -> no value accepted (most specific / null)
    a specific value (e.g. 'Sunny') -> exactly that value required

This is the classic "EnjoySport" dataset from Tom Mitchell's
Machine Learning textbook.

Run: python3 candidate_elimination.py
"""

import pandas as pd

# ----------------------------------------------------------------------
# 1. TRAINING DATA
# ----------------------------------------------------------------------
data = {
    "Sky":      ["Sunny", "Sunny", "Rainy", "Sunny"],
    "AirTemp":  ["Warm",  "Warm",  "Cold",  "Warm"],
    "Humidity": ["Normal","High",  "High",  "High"],
    "Wind":     ["Strong","Strong","Strong","Strong"],
    "Water":    ["Warm",  "Warm",  "Warm",  "Cool"],
    "Forecast": ["Same",  "Same",  "Change","Change"],
    "EnjoySport": ["Yes", "Yes",   "No",    "Yes"],
}
df = pd.DataFrame(data)

print("=" * 70)
print("TRAINING EXAMPLES")
print("=" * 70)
print(df.to_string(index=False))

attributes = df.columns[:-1].tolist()   # all columns except the target
target_col = df.columns[-1]
n_attrs = len(attributes)

X = df[attributes].values.tolist()
y = df[target_col].values.tolist()


def more_general(h1, h2):
    """True if h1 is more-or-equally general than h2, attribute-wise."""
    for a, b in zip(h1, h2):
        if a == "?":
            continue
        if a == "phi":
            return False
        if a != b and b != "phi":
            return False
    return True


def consistent(hypothesis, instance):
    """True if hypothesis is satisfied by the instance."""
    for h, x in zip(hypothesis, instance):
        if h != "?" and h != x:
            return False
    return True


def minimal_generalizations(h, instance):
    """Smallest generalizations of h that cover the new positive instance."""
    new_h = list(h)
    for i in range(len(h)):
        if h[i] == "phi":
            new_h[i] = instance[i]
        elif h[i] != instance[i]:
            new_h[i] = "?"
    return [tuple(new_h)]


def minimal_specializations(h, domains, instance):
    """Smallest specializations of h that exclude the negative instance."""
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if val != instance[i]:
                    new_h = list(h)
                    new_h[i] = val
                    results.append(tuple(new_h))
        elif h[i] != "phi":
            pass  # already specific on this attribute; cannot narrow further
    return results


# domain of possible values for each attribute (used when specializing G)
domains = [sorted(set(df[col])) for col in attributes]

# ----------------------------------------------------------------------
# 2. INITIALIZE S AND G
# ----------------------------------------------------------------------
S = [tuple(["phi"] * n_attrs)]          # most specific boundary
G = [tuple(["?"] * n_attrs)]            # most general boundary

print("\n" + "=" * 70)
print("INITIALIZATION")
print("=" * 70)
print("S0 =", S)
print("G0 =", G)

# ----------------------------------------------------------------------
# 3. PROCESS EACH TRAINING EXAMPLE
# ----------------------------------------------------------------------
for idx, (instance, label) in enumerate(zip(X, y), start=1):
    print("\n" + "-" * 70)
    print(f"Example {idx}: {instance}  ->  {label}")
    print("-" * 70)

    if label == "Yes":  # ---------- POSITIVE EXAMPLE ----------
        # Remove from G any hypothesis inconsistent with the instance
        G = [g for g in G if consistent(g, instance)]

        new_S = []
        for s in S:
            if consistent(s, instance):
                new_S.append(s)
            else:
                for gen in minimal_generalizations(s, instance):
                    # keep only if some member of G is more general than gen
                    if any(more_general(g, gen) for g in G):
                        new_S.append(gen)
        # remove hypotheses in S that are more general than another in S
        S = [s for s in new_S if not any(
            s != other and more_general(other, s) for other in new_S)]

    else:  # ---------------------- NEGATIVE EXAMPLE ----------------------
        # Remove from S any hypothesis consistent with the (negative) instance
        S = [s for s in S if not consistent(s, instance)]

        new_G = []
        for g in G:
            if not consistent(g, instance):
                new_G.append(g)
            else:
                for spec in minimal_specializations(g, domains, instance):
                    # keep only if some member of S is more specific than spec
                    if any(more_general(spec, s) for s in S):
                        new_G.append(spec)
        # remove hypotheses in G that are more specific than another in G
        G = [g for g in new_G if not any(
            g != other and more_general(g, other) for other in new_G)]

    print("S boundary:", S)
    print("G boundary:", G)

# ----------------------------------------------------------------------
# 4. FINAL VERSION SPACE
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL VERSION SPACE (all hypotheses consistent with training data)")
print("=" * 70)
print("\nMost Specific Boundary (S):")
for s in S:
    print("  ", s)

print("\nMost General Boundary (G):")
for g in G:
    print("  ", g)

if S == G:
    print("\nS == G  ->  the version space has CONVERGED to a single hypothesis:")
    print("  ", S[0])
else:
    print("\nEvery hypothesis 'h' with S <= h <= G (more general-than-or-equal-to S,")
    print("more specific-than-or-equal-to G) is consistent with all training examples.")
