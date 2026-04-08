

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
- Multifactorial that have influence
- Genes: If in family, then correlation in descendance high
  - Probably how you abbauen cholesterole and fat -> Haengt vom Stoffwechsel ab -> Wie gut die abgebaut werden oder ob die irgendwo haengen bleiben
- Herzkranzgefaesse: Wie aufgebaut? Wie elastisch? Wie glatt innere Wand (endothel)
- Meistens eine Form von Herzinfarkt (Endgefaesse verstopfen)
  - Kein starres Rohr: Elastizitaet wichtig und auch wie rau oberflaeche ist -> Bleibt mehr haengen -> verstopft schneller
- Konsum (von klein an) / Detritus:
  - Fetthaltige Lebensmittel -> Da kommste auf Cholesterol
  - Tabak und Alkohol
  - Zucker?
- Systolic blood pressure: Auswurf vom Herzen -> Wsl Einfluss -> Muss durch gesamten grossen Kreislauf -> damit selber versorgen -> Wenn zu schwach, Gefaesse zu eng, dann kann es sich selber nicht gut versorgen.
- ldl: Ein Tri-Glycerit (Fett). Gibt high and low density. Faktor fuer Abbau auf Produkt. Gibt einen der Boese ist und einer der Gut ist. War LDL oder HDL das problematische? Da waren auch genetische Komponenten dazu.
- Alter: Gefaesse werden mehr zugeschlammt, Endotel wird sporoeser (rauher, weniger glatt), altes Rohr auch poroeser. Elastizitaet nimmt auch ab. Muskuloes eventuell auch (Zellen werden aelter), aber nicht sicher.
- Type-A behavior: 
  - Stress -> Herz schlaegt flach und schnell und versorgt sich selbst nicht so gut -> Weniger durchfluss durh gefaesse, mehr Ablagerungen -> Mehr Risiko. 
  - Stress -> Cortisol
  - Stress -> Leute ernaehren sich dann auch nicht mehr so gut, weil sie sich nicht mehr die Zeit nehmen.
- Obesity: Ueber BMI definiert (?) -> Wie fettleibig man ist -> Mehr Fettgewebe -> Automatisch auf Verteilung -> haengt nah mit Adiposity zusammen
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


# Things to try
- We can play with expert knowledge:
- expert_knowledge: pgmpy.estimators.ExpertKnowledge (default: None). Expert knowledge about the causal structure. This can include:
  - forbidden_edges: Edges that should not be present in the final model
  - required_edges: Edges that must be present in the final model (can be removed during pruning)
  - temporal_order: The temporal ordering of variables. Note that explicit orientations
  specified in the 'orientations' parameter will override this temporal ordering.


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

- I'm having issues with a frozenset error. It occurs once you enable the expert knowledge. But expert knowledge is quite powerful. I left it in a state where it works. I've also tried the newer, not outdated version of pgmpy, but the error remains.