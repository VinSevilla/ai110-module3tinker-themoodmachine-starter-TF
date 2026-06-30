# TF Guide — Mood Machine Assignment

This guide is for teaching assistants running the Mood Machine lab. It covers the full premise of the assignment, what students are expected to build, where they commonly get stuck, how to redirect them, and what good work looks like at each stage.

---

## Assignment Premise

Students build a rule-based mood classifier from scratch in Python. The system reads short text posts — social media style — and predicts whether the mood is positive, negative, neutral, or mixed.

The assignment is split into four main parts:

| Part | File | What students do |
|---|---|---|
| 1 | `dataset.py` | Expand the labeled dataset with realistic posts |
| 2 | `mood_analyzer.py` | Implement `preprocess`, `score_text`, `predict_label` |
| 3 | `mood_analyzer.py` | Run evaluation, observe failures |
| 4 | `model_card.md` | Document the model, its limits, and ethical concerns |
| Extension | `ml_experiments.py` | Compare against a scikit-learn ML model |

**The central learning goal is not accuracy.** It's understanding *why* a model makes the predictions it does. A student who gets 50% accuracy and can explain every failure understands more than a student who patches their word list to hit 80% without knowing why.

---

## Part 1 — dataset.py

### What students are building

`SAMPLE_POSTS` is a list of short text strings. `TRUE_LABELS` is a parallel list of mood labels. Index alignment is the only contract — `SAMPLE_POSTS[i]` is labeled by `TRUE_LABELS[i]`.

Students must add 5–10 new posts and a matching label for each one. The starter has 6 posts. They should end up with at least 11.

### What good work looks like

- Posts cover multiple styles: clean literal language, slang, emojis, negation, sarcasm, ambiguous feelings
- Labels are justified — students can explain *why* they chose each one
- Some posts are edge cases that are genuinely hard to label
- `len(SAMPLE_POSTS) == len(TRUE_LABELS)` always holds

### Common mistakes

**Length mismatch** — the most frequent error in this section. A student adds a post but forgets the label, or adds two labels for one post. This causes a crash elsewhere.

Quick check to share with students:
```python
python3 -c "from dataset import SAMPLE_POSTS, TRUE_LABELS; print(len(SAMPLE_POSTS), len(TRUE_LABELS))"
```

**Only adding easy posts** — students default to clean, literal sentences like "I feel sad today." Push them toward posts that are harder: "lol i'm fine 😐", "not bad i guess", "this is literally the worst best day."

**Labeling confusion around `mixed` vs `neutral`** — students often conflate these two.
- `neutral` = no strong signal in either direction. "This is fine." "Just another day."
- `mixed` = signals pointing in *both* directions at once. "Tired but hopeful." "Failed my test but honestly needed the break."

A post can't be both. If a student is unsure, ask: *is there emotional content pointing in two directions, or is there just no strong emotion at all?*

**Labeling based on literal words instead of intent** — for sarcasm, students sometimes label the literal reading. "I absolutely love getting stuck in traffic" gets labeled `positive` because they see "love." Ask them: *what would the person who wrote this actually be feeling?*

### Discussion questions for this section

- Which post was hardest to label? Would someone else label it the same way?
- What makes a label ambiguous?
- If you and a classmate disagreed on a label, who is "right"?

---

## Part 2 — mood_analyzer.py

Students implement three methods in sequence. Each feeds the next.

```
preprocess(text) → list of tokens
score_text(text) → int
predict_label(text) → str
```

### preprocess

**What it should do:** lowercase, split on spaces, strip punctuation from token edges, drop empty strings.

**What good work looks like:**
```python
preprocess("good, not bad!")
# → ["good", "not", "bad"]

preprocess("I love this 😅")
# → ["i", "love", "this", "😅"]
```

Key behaviors to verify:
- `"good,"` becomes `"good"` — punctuation stripped from edges
- `"😅"` stays as `"😅"` — emojis are not in `string.punctuation`
- `"..."` disappears — becomes empty string, then filtered
- Contractions like `"i'm"` keep their apostrophe — it's internal, not on the edge

**Common mistakes:**

*Forgetting to import `string`* — students add `t.strip(string.punctuation)` without importing it. The error is `NameError: name 'string' is not defined`. The fix is `import string` at the top of the file.

*Not filtering empty strings* — after stripping, a token that was purely punctuation (like `"..."`) becomes `""`. If they don't filter it, the empty string ends up in the token list and breaks scoring.

*Over-engineering too early* — some students want to handle contractions, normalize "soooo→so", and detect every emoji before moving on. Redirect them: get it working simply first, then improve if time allows.

*Splitting on `""` (empty string) instead of spaces* — produces individual characters. Remind them: `.split()` with no argument splits on whitespace, which is what they want.

### score_text

**What it should do:** call `preprocess`, loop over tokens, +1 for positive words, -1 for negative words, return total.

**What good work looks like:**
```python
score_text("I love this class")   # → 1
score_text("Today was terrible")  # → -1
score_text("This is fine")        # → 0
```

**Common mistakes:**

*Leaving `pass` and not returning anything* — `pass` implicitly returns `None`. Downstream, `predict_label` does `if score > 0` which raises `TypeError: '>' not supported between instances of 'NoneType' and 'int'`. The error message is confusing to students because they implemented `predict_label` and it looks fine. Teach them to check whether `score_text` is actually returning a value.

*Using `if` instead of `elif` for the negative check* — a word that somehow appears in both lists gets counted twice. Not a common bug with the starter lists, but it becomes one when students expand the word lists.

*Not calling `self.preprocess(text)` first* — some students write the loop over `text.split()` directly and skip preprocessing. Their tokenization won't match the word lists because of punctuation and case differences.

*Negation applied too broadly* — students sometimes negate the entire rest of the sentence after seeing "not" instead of just the immediately following word. "I am not happy and also very excited" should score `happy` as negated but `excited` as genuinely positive. Ask them: *how far does "not" usually reach in a sentence?*

**The negation implementation to share if students are stuck:**

```python
negation_words = {"not", "never", "don't", "didn't", "doesn't", "can't", "isn't", "wasn't"}

for i, token in enumerate(tokens):
    preceding = tokens[i - 1] if i > 0 else ""
    negated = preceding in negation_words

    if token in self.positive_words:
        score += -1 if negated else 1
    elif token in self.negative_words:
        score += 1 if negated else -1
```

### predict_label

**What it should do:** call `score_text`, map the result to a label string.

Baseline mapping:
```
score > 0  → "positive"
score < 0  → "negative"
score == 0 → "neutral"
```

**Common mistakes:**

*Hardcoding the return* — a student writes `return "positive"` instead of using the score. Their model looks like it works on one or two examples, then produces the same label for everything. Ask them to run it on a clearly negative post.

*Forgetting to call `score_text`* — they define `score = 0` inside the method without calling `self.score_text(text)`.

*Wanting to add `mixed` here but not knowing how* — this is a good teaching moment. Explain that a raw score loses information: a score of 0 could mean "nothing matched" or "things cancelled." To support `mixed`, they'd need to track positive and negative hit counts separately inside `score_text` and pass that information up. The single-number score doesn't carry enough signal.

### Running the model

To test the full pipeline:
```
python3 main.py
```

To evaluate with side-by-side comparison (useful for debugging):
```python
python3 -c "
from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS
a = MoodAnalyzer()
for post, true in zip(SAMPLE_POSTS, TRUE_LABELS):
    pred = a.predict_label(post)
    mark = '✓' if pred == true else '✗'
    print(f'{mark} {true:<10} {pred:<10} {post}')
"
```

---

## Part 3 — Evaluation and Analysis

Students run the model against `TRUE_LABELS` and analyze where it fails. This is the most important intellectual part of the lab.

### What good analysis looks like

A student who just says "the model is wrong sometimes" has not completed this section. Good analysis:
1. Picks a specific incorrect prediction
2. Traces through `preprocess` → `score_text` → `predict_label` token by token
3. Identifies *why* the model made the wrong call
4. Decides whether the error is fixable within the current approach or is a structural limitation

### Common failure patterns to help students recognize

**Sarcasm** — the model finds a positive keyword (`love`, `great`) and scores it positive, ignoring that the surrounding context is negative. Example: `"great, another Monday 🙄"` → predicts `positive`, true label `negative`. This is a structural limitation. Keyword matching cannot resolve sarcasm because sarcasm is defined by the relationship between what words literally mean and what context implies.

**Vocabulary gap / slang** — tokens like `"lowkey"`, `"highkey"`, `"vibing"`, `"no cap"` don't appear in the word lists and score 0. Example: `"highkey obsessed with this playlist rn 🎧❤️"` → predicts `neutral` because no tokens match. This *is* fixable — add the words to the word lists. But it doesn't scale: you can't anticipate every slang term.

**Subtle negative language** — posts that express low mood without using the exact negative words in the list. Example: `"the party was okay i guess... felt kinda invisible tbh"` → predicts `neutral` because `"invisible"`, `"kinda"`, `"okay"` are all unknown. The emotional signal is real but the vocabulary doesn't capture it.

**Emoji signals ignored** — emojis like 🙄 (sarcasm), 💀 (hyperbole/dark humor), 😭 (often positive crying in context) carry strong sentiment that the model can't read. They score 0 every time.

**`mixed` is unpredictable** — the model has no path to predict `mixed`. Any post with that label is a guaranteed error. Students should recognize this as a *design* limitation, not a vocabulary limitation.

### Useful debugging tool

The `explain` method in `MoodAnalyzer` shows which tokens fired and what the final score was:
```python
from mood_analyzer import MoodAnalyzer
a = MoodAnalyzer()
print(a.explain("I am not happy about this"))
# Score = -1 (positive: [], negative: [])  ← after negation fix
```

**Note for TFs:** `explain` has its own scoring loop that does not apply negation — it will show a different score than `score_text` if the student has implemented negation. This is a known discrepancy. It's worth pointing out as an example of what happens when logic is duplicated across two methods.

---

## Part 4 — model_card.md

Students fill in a structured template covering: how the model works, data description, evaluation results, limitations, and ethical considerations.

### What good work looks like

- Specific examples from their actual evaluation run, not generic statements
- Limitations are illustrated with real predictions, not just described
- The ethical section connects to the actual failure modes they observed
- "Mixed" label and sarcasm are both addressed

### Common problems

**Too vague** — "the model struggles with sarcasm" without showing an example. Ask them to name the post, what it predicted, what the true label was, and *why the model was wrong at the token level.*

**Skipping evaluation results** — some students describe the model without reporting what accuracy they observed. They need to run `main.py` and include the number.

**Treating 100% ML accuracy as a success** — if they ran `ml_experiments.py`, the ML model will likely score 100% on training data. Students often present this uncritically. Key question to ask: *was the model evaluated on data it had already seen?* With 18 examples and no held-out test set, 100% training accuracy means the model memorized the data, not that it learned something general.

**Ethical section is too abstract** — students write "AI can be biased" without connecting it to their specific model. Push them to be concrete: *who wrote the posts in your dataset? What kinds of language did you include? What kinds did you leave out? Who would this model misread?*

---

## ML Extension — ml_experiments.py

### What it does

`train_ml_model` uses `CountVectorizer` (bag of words) and `LogisticRegression` from scikit-learn. It trains on `SAMPLE_POSTS` and `TRUE_LABELS`, then evaluates on the same data.

Run it with:
```
python3 ml_experiments.py
```

### Key teaching points

**Training accuracy vs. generalization** — with ~18 examples, logistic regression will almost always hit 100% on training data. This is memorization. There is no held-out test set, so the number means nothing for unseen posts. This is one of the main things students should take away from the extension.

**The ML model can predict `mixed`** — because it saw `mixed` labels during training and learned to associate certain word patterns with that label. The rule-based model cannot do this at all. This is a genuine advantage of learned models.

**Label sensitivity** — with a tiny dataset, every label decision influences the model heavily. If a student mislabels two or three posts, the ML model may learn spurious patterns. This is a good entry point for discussing data quality.

**The ML model is a black box** — unlike the rule-based model, students can't trace *why* the ML model made a decision. They can inspect feature weights if they know how, but there's no `explain()` equivalent. This tradeoff — interpretability vs. flexibility — is worth discussing explicitly.

### Common issue

*scikit-learn not installed* — `ModuleNotFoundError: No module named 'sklearn'`. Fix:
```
pip install scikit-learn
```

---

## Quick Reference: Error Messages and Fixes

| Error | Likely cause | Fix |
|---|---|---|
| `TypeError: '>' not supported between NoneType and int` | `score_text` returns `None` (still has `pass`) | Implement `score_text` to return an `int` |
| `NameError: name 'string' is not defined` | `import string` missing | Add `import string` at top of `mood_analyzer.py` |
| `AssertionError` or wrong count when checking lengths | `SAMPLE_POSTS` and `TRUE_LABELS` length mismatch | Count rows in both lists; add or remove to align |
| `ModuleNotFoundError: No module named 'sklearn'` | scikit-learn not installed | `pip install scikit-learn` |
| Every post predicts the same label | `predict_label` is hardcoded, or `score_text` always returns 0 | Check that `score_text` loops over tokens and checks word lists |
| Negation not working | Preceding token check is off-by-one | Print `tokens[i-1]` for each `i` to verify the window |

---

## Concepts Students Should Be Able to Explain by the End

These are good exit-ticket or check-in questions:

1. **What is tokenization and why does it matter?** If "good," stays "good," does it match the word list?
2. **What does a score of 0 mean?** Is it always neutral? Can it mean something else?
3. **Why can't the rule-based model predict `mixed`?** What would you need to change to support it?
4. **What is negation and how does the window work?** Does "not" negate the next word? The whole sentence?
5. **Why does sarcasm break keyword matching?** What would a model need to understand in order to detect it?
6. **Why is 100% ML accuracy on training data suspicious?** What's the difference between memorizing and learning?
7. **Who is this model optimized for?** What kinds of language or people would it misread?

---

## TF Notes on Grading / Feedback

- The quality of the model card matters more than accuracy. A student at 50% with a well-reasoned card has done better conceptual work than one at 70% who can't explain their failures.
- Look for students who blame the model without analyzing it. "It just got it wrong" is not analysis. Ask: *which token caused this? What did the model actually see?*
- The `mixed` label is guaranteed to be wrong for the rule-based model. Don't penalize students for this — reward them if they identify it as a structural limitation rather than a bug.
- Sarcasm failures are expected. Reward students who recognize that the failure is not fixable with a bigger word list — it requires understanding context.
- If a student's ML accuracy is exactly 100%, make sure they understand why that doesn't mean the model is good.
