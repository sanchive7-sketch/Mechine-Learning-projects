"""
FIND-S Algorithm
------------------
Finds the maximally SPECIFIC hypothesis that is consistent with all
the POSITIVE training examples (Find-S ignores negative examples
entirely -- it only ever generalizes, never specializes).

Hypothesis representation (conjunction of constraints per attribute):
    '?'    -> any value accepted (fully general for that attribute)
    'phi'  -> no value accepted yet (initial / null constraint)
    value  -> exactly that value required

Algorithm:
    1. Initialize h to the most specific hypothesis: (phi, phi, ..., phi)
    2. For each POSITIVE training example x:
         For each attribute a_i:
             if h[i] == 'phi'        -> h[i] = x[i]
             elif h[i] != x[i]       -> h[i] = '?'
             else                    -> leave h[i] unchanged
       (negative examples are skipped completely)
    3. Output h -- this is the most specific hypothesis consistent
       with all positive examples.

Run: python3 find_s.py
"""

import pandas as pd

# ----------------------------------------------------------------------
# 1. TRAINING DATA (same EnjoySport dataset used for Candidate-Elimination,
#    so the two demos can be compared directly)
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

attributes = df.columns[:-1].tolist()
target_col = df.columns[-1]
n_attrs = len(attributes)

X = df[attributes].values.tolist()
y = df[target_col].values.tolist()

# ----------------------------------------------------------------------
# 2. INITIALIZE h TO THE MOST SPECIFIC HYPOTHESIS
# ----------------------------------------------------------------------
h = ["phi"] * n_attrs

print("\n" + "=" * 70)
print("INITIALIZATION")
print("=" * 70)
print("h0 =", tuple(h))

# ----------------------------------------------------------------------
# 3. PROCESS EACH TRAINING EXAMPLE
# ----------------------------------------------------------------------
for idx, (instance, label) in enumerate(zip(X, y), start=1):
    print("\n" + "-" * 70)
    print(f"Example {idx}: {instance}  ->  {label}")
    print("-" * 70)

    if label != "Yes":
        print("Negative example -> IGNORED by Find-S")
        print("h unchanged:", tuple(h))
        continue

    for i in range(n_attrs):
        if h[i] == "phi":
            h[i] = instance[i]
        elif h[i] != instance[i]:
            h[i] = "?"
        # else: h[i] already matches this attribute value -> keep as is

    print("Positive example -> generalize h")
    print("h updated:", tuple(h))

# ----------------------------------------------------------------------
# 4. FINAL HYPOTHESIS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL MOST SPECIFIC HYPOTHESIS (Find-S output)")
print("=" * 70)
print(tuple(h))

print("""
Note on "all hypotheses consistent with the training examples":
Find-S outputs only ONE hypothesis -- the most specific member of the
version space -- because it only ever generalizes to accommodate
positive examples and completely ignores negative examples.

It is provably equal to the S boundary produced by the
Candidate-Elimination algorithm (compare with candidate_elimination.py).
It does NOT, by itself, describe the full set of consistent hypotheses:
that requires also computing the G boundary (most general hypotheses
consistent with the negatives), which is exactly what
Candidate-Elimination adds on top of Find-S's idea.
""")
