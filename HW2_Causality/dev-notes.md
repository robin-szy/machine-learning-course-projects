

## Explanation variables

- sbp: systolic blood pressure
- tobacco: cumulative tobacco (kg)
- ldl: low density lipoprotein cholesterol
- adiposity: the amount or distribution of fat tissue (adipose tissue) in the body
- famhist: family history of heart disease (Present=1, Absent=0)
- typea: type-A behavior
- obesity: medical condition characterized by excessive body fat that may impair health
- alcohol: current alcohol consumption
- age: age at onset
- chd: coronary heart disease (yes=1 or no=0)

### What is type-A behavior?

Source: https://en.wikipedia.org/wiki/Type_A_and_Type_B_personality_theory

A person with Type A behavior is typically:
- Highly competitive
- Strong sense of urgency (“always in a rush”)
- Impatient
- Prone to stress and hostility
- Often high-achieving workaholics

In contrast, Type B personalities are more relaxed and less stress-driven.


# A-priori thoughts and expectations

ChatGPT:
- **Direct causes of CHD**:`ldl, sbp, tobacco, obesity, age, famhist`
- **Upstream causes (confounders/mediators)**:`typea, alcohol`
- **Redundant / overlapping**:`adiposity, obesity`

Robin's girlfriend (vet):

- Multifactorial factors that have influence  
- Genes: If in the family, then correlation in descendants is high  
  - Probably how you break down cholesterol and fat → depends on metabolism → how well they are broken down or whether they get stuck somewhere  
- Coronary arteries: How are they structured? How elastic? How smooth is the inner wall (endothelium)  
- Mostly a form of heart attack (end vessels get blocked)  
  - Not a rigid pipe: elasticity is important and also how rough the surface is → more sticks → clogs faster  
- Consumption (from an early age) / detritus:  
  - Fatty foods → that’s how you get to cholesterol  
  - Tobacco and alcohol  
  - Sugar?  
- Systolic blood pressure: Output from the heart → probably has an influence → has to go through the entire large circulation → has to supply itself → if too weak, vessels too narrow, then it cannot supply itself properly  
- LDL: A triglyceride (fat). There is high and low density. Factor for breakdown into product. There is one that is bad and one that is good. Was LDL or HDL the problematic one? There were also genetic components involved.  
- Age: Vessels become more clogged, endothelium becomes more porous (rougher, less smooth), old pipe also more porous. Elasticity also decreases. Possibly also muscular (cells get older), but not certain.  
- Type-A behavior:  
  - Stress → heart beats shallow and fast and does not supply itself as well → less flow through vessels, more deposits → higher risk  
  - Stress → cortisol  
  - Stress → people also don’t eat as well anymore because they don’t take the time  
- Obesity: Defined via BMI (?) → how obese someone is → more fat tissue → automatically affects distribution → closely related to adiposity  
- Adiposity: Basically the same as obesity.  


# Other notebooks

- **Chain (mediator)**:  
  True causal effect from `X1` to `X3` via `X2` (`X1 → X2 → X3`).  
  - Marginal: corr ≈ **0.5–0.6**, ATE ≈ **0.5–0.6**  
  - Conditional on `X2`: corr ≈ **0**  
  → Controlling for `X2` **removes the real effect** (blocks the path)  

- **Fork (confounder)**:  
  No causal effect between `X1` and `X3`, both caused by `X2` (`X2 → X1`, `X2 → X3`).  
  - Marginal: corr ≈ **0.2–0.3**, ATE ≈ **0.2–0.3**  
  - Conditional on `X2`: corr ≈ **0**  
  → Controlling for `X2` **removes spurious correlation** (correct adjustment)  

- **Collider (inverted fork)**:  
  No causal effect between `X1` and `X3`, both influence `X2` (`X1 → X2 ← X3`).  
  - Marginal: corr ≈ **0**  
  - Conditional on `X2`: corr ≈ **0.2–0.4** (non-zero appears)
  → Controlling for `X2` **creates fake correlation** (collider bias)

- **Backdoor adjustment (confounding control)**:  
  Estimate causal effect of `X → Y` in presence of a confounder `Z` (`Z → X`, `Z → Y`).  
  - Without adjustment: corr ≈ **0.3–0.5** (biased), ATE ≠ true effect  
  - Conditional on `Z`: corr ≈ **true causal effect** (bias removed)  
  → Adjusting for `Z` **blocks the backdoor path** (`X ← Z → Y`) and recovers the correct causal effect

  - **Key rule (Backdoor Criterion)**:  
    Choose a set `Z` such that:  
    - It blocks all backdoor paths from `X` to `Y`  
    - It does **not include descendants of `X`**
    - Don't control for mediators or colliders


# Things I've modified
- I disabled the binning into 4 bins. Binning involves expert knowledge we don't have at that point. Here, it is data driven (oh nice, 4 bins is convenient). Yes, but why the threshold? Doesn't make sense to do right now -> Out.
  - We can do it, but thresholds should make sense!
- I've decreased the significance_level from 0.2 to 0.01. The higher it is, the less edges are kept. 
  - Basically: p_value > significance_level -> accept independence -> remove edge. Null hypothesis: Variables independent.
  - See the following excerpt from docs:
    - The statistical tests use this value to compare with the p-value of the test to decide whether the tested variables are independent or not. Different tests can treat this parameter differently:
      - Chi-Square: If p-value > significance_level, it assumes that the independence condition satisfied in the data. 
      - pearsonr: If p-value > significance_level, it assumes that the independence condition satisfied in the data.
- Included expert knowledge: Made age and genetics upstream because none of the variables can influence them. Downstream I put the disease. I also forbid edges out of the disease, but took it out again, as chd did not occur in the map anymore afterwards. With the hierarchy it works quite fine, though.
- I got a lot of future warnings. I've updated the code to the newest library version.
- Obesity and Adiposity are quite similar. I removed obesity.

- I had issues with a frozenset error. It occurs once you enable the expert knowledge. But expert knowledge is quite powerful. I left it in a state where it works. I've also tried the newer, not outdated version of pgmpy, but the error remains.
- I was able to fix it. It is an actual bug in the library of PC itself. It's an easy fix. Just replace in the library file at .venv/lib/python3.12/site-packages/pgmpy/estimators/PC.py the following lines:
```python
for X, Y in permutations(sorted(pdag.nodes()), 2):
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        if (temporal_ordering == dict()) or (
                            (temporal_ordering[Z] >= temporal_ordering[X])
                            and (temporal_ordering[Z] >= temporal_ordering[Y])
                        ):
                            pdag.remove_edges_from([(Z, X), (Z, Y)])
```

by this here:
```python
for X, Y in permutations(sorted(pdag.nodes()), 2):
    if not skeleton.has_edge(X, Y):
        sep_set = separating_sets.get(frozenset((X, Y)))    # Fix frozen set bug
        if sep_set is None:
            continue

        for Z in set(skeleton.neighbors(X)) & set(
                skeleton.neighbors(Y)):
            if Z not in sep_set:
                if (temporal_ordering == dict()) or (
                        (temporal_ordering[Z] >= temporal_ordering[X])
                        and (temporal_ordering[Z] >= temporal_ordering[
                    Y])
                ):
                    pdag.remove_edges_from([(Z, X), (Z, Y)])
```

Then restart the Kernel, and it should work.


## Things to try:

- Optional, because takes a lot of time (and we don't want to exaggerate): Use correlation_score, structure_score, and log_likelihood_score for comparing the models? Or so. Right now, we just look at the graph, but we don't really know how confident we can be