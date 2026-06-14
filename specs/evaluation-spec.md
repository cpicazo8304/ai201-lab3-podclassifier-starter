# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[blank — write out the accuracy formula in plain English.
 What counts as "correct"? What do you divide by?]
```
Correct means that the prediction_normalized == ground_truth_normalized

number of correct / total examples

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. ...
 2. ...
 3. ...]
```
-first check length of both lists equal
-if so, loop through each of the predictions and ground truth
-Check if they equal each other normalized, if so, increment correct count
-once done looping, divide number of correct by total examples
---

**Edge case — what if both lists are empty?**

```
[blank — what should the function return? Why?]
```
If both lists are empty, return 0.0.

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[blank — what does compute_accuracy() return for these inputs? Show your work.]
```
it would return 2/4 so 0.5 since the first two predictions are correct and the final two are not.

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[blank — be precise. When does an episode count as correctly classified
 for the "interview" class, for example?]
```
Correct means that the groud truth is for a specific class and the prediction has the same label as that specific class.

For example "interview" class, the ground truth has to be "interview" for it to qualify it as correct or not correct for the specific class. So, you have to be working with the specific class before determining if it is right or wrong.

---

**What does "total" mean for a given class?**

```
[blank — is "total" the total number of predictions, or something more specific?]
```
total is the total number of ground truths that are in the given class.

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]
```

1. Initialize different total variables for each class and different num correct variables for each class.
2. loop through the lists, check which ground truth is being used and increment that corresponding total variable
3. check if the prediction equals ground truth and increment the corresponding num correct variables
4. once finished looping, calculate each of the four accuracies if applicable

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]
```
It should be 0.0.

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

[blank — fill in the per-class results table below]

label       correct  total  accuracy
----------  -------  -----  --------
interview   [1]  [1]  [1]
solo        [1]  [2]  [0.5]
panel       [1]  [1]  [1]
narrative   [0]  [1]  [0]
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
