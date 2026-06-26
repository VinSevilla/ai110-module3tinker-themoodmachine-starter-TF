# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I used the rule based model only.

**Intended purpose:**  
The model reads a short piece of text — like a social media post or a quick message — and classifies the mood as positive, negative, or neutral. The goal was to build something simple that I could trace step by step and understand exactly why it gave each answer.

**How it works (brief):**  
The model works in three steps. First, it preprocesses the text: lowercases everything, splits on spaces, and strips punctuation from the edges of each token so that words like `"good,"` don't get missed. Then it scores the text by looping through the tokens and adding +1 for any word that appears in a positive word list and -1 for any word in a negative word list. I also added a negation rule: if the token right before a sentiment word is something like "not", "never", or "don't", the score flips — so "not happy" scores -1 instead of +1. Finally, the score gets mapped to a label: above 0 is positive, below 0 is negative, and exactly 0 is neutral.



## 2. Data

**Dataset description:**  
The starter dataset had 6 posts. I expanded it to 18 total by adding 12 new examples across four rounds. I tried to cover language styles that the original posts didn't have: slang like "lowkey" and "highkey", emojis like 😅 and 🙄, sarcasm, ambiguous feelings, and negation.

**Labeling process:**  
I labeled each post myself based on what I thought the author actually meant, not what the words literally said. That distinction mattered a lot for sarcasm and mixed-feeling posts.

Some posts were genuinely hard to label:

- `"lowkey been crying all day but at least the pizza was good 😅"` — is this negative because of crying, or mixed because something good happened? I went with `mixed`, but I could see an argument for `negative`.
- `"failed my exam but tbh im not even mad anymore 💀"` — the outcome is bad, but the emotional state sounds resigned rather than upset. I labeled it `mixed`.
- `"honestly don't know how I feel rn 😶"` — this one is genuinely ambiguous by design. I labeled it `neutral` because the author isn't expressing a clear direction.

**Important characteristics of your dataset:**  
- Contains slang: "lowkey", "highkey", "no cap", "tbh", "bestie"
- Contains emojis: 😅, 🔥, 😐, 😭💕, 💀, 🤷, 🎧❤️, 😶, 🙄
- Includes sarcasm: at least three posts where the literal words say the opposite of what the author means
- Includes negation: "I am not happy about this", "not bad, actually kind of fun"
- Includes mixed-emotion posts where a single label feels like a simplification
- Most posts are very short (under 15 words)

**Possible issues with the dataset:**  
The dataset is small — 18 posts is not enough to draw real conclusions about accuracy. It also skews toward the kinds of posts I thought of, which are mostly casual and playful. I don't have any posts that express serious distress, anger, or more formal language. The `mixed` label appears in 3 posts but the model can never actually predict it, so those 3 are guaranteed errors from the start.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
The positive and negative word lists each have 10 words from the starter. Every token that matches a positive word adds 1 to the score. Every token that matches a negative word subtracts 1. I used `elif` for the negative check so a word can't count as both positive and negative at the same time.

For negation, I keep a small set of negation words: `{"not", "never", "don't", "didn't", "doesn't", "can't", "isn't", "wasn't"}`. Before scoring a token, I check whether the token immediately before it is a negation word. If it is, the score contribution flips — a positive word becomes -1 and a negative word becomes +1.

For the label threshold, I kept the simplest possible mapping: score > 0 → positive, score < 0 → negative, score == 0 → neutral. I considered requiring a stronger score like >= 2 for positive, but most posts in my dataset only score 1 or -1 because the word lists are short. A stricter threshold would have pushed almost everything to neutral, which wasn't useful.

There is no emoji handling and no slang handling — those tokens go unrecognized and contribute 0.

**Strengths of this approach:**  
- It's completely transparent. I can look at any prediction and trace exactly which tokens caused it.
- It handles negation for common patterns. "I am not happy about this" went from scoring +1 (wrong) to -1 (correct) after I added negation logic.
- It's fast and doesn't require any training data.
- For clean, literal language with words that appear in the word lists, it tends to get the direction right.

**Weaknesses of this approach:**  
- It has no concept of context. Each token is scored independently, so the relationship between words doesn't matter.
- It can't detect sarcasm. More on this in the Limitations section.
- It can't predict the `mixed` label at all. The only path to `mixed` would be tracking positive and negative hit counts separately, not just the final score — a score of 0 could mean "nothing matched" or "things cancelled", and the model can't tell the difference.
- Most slang and emojis score 0 because they're not in the word lists.

## 4. How the ML Model Works (if used)

**Features used:**  
`CountVectorizer` from scikit-learn converts each post into a bag-of-words vector — every unique word across the dataset becomes a column, and each post becomes a row of word counts. The model doesn't know word order or context, just which words appeared and how many times.

**Training data:**  
The same 18 posts and labels from `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
The ML model scored 18/18 on the training data — 100% accuracy compared to 9/18 for the rule-based model. But this number is misleading because the model is being evaluated on the exact data it trained on. With only 18 examples and a logistic regression, it almost certainly memorized the training set rather than learning anything that would generalize.

The sensitivity to labels was real and a little surprising. Because the dataset is so small, every label carries a lot of weight. If I had labeled `"Feeling tired but kind of hopeful"` as `negative` instead of `mixed`, the model would have learned something different about the word "hopeful" — and that change would probably ripple into predictions for other posts containing similar words.

**Strengths and weaknesses:**  
The most notable strength is that the ML model handled things the rule-based model completely missed. It correctly predicted `mixed` for all three mixed-label posts, which the rule-based model can never do. It also correctly predicted `negative` for the three sarcasm posts — not because it understood sarcasm, but because it memorized which training examples had those labels.

The major weakness is that 100% training accuracy on 18 examples is not a real measure of anything. If a genuinely new sarcastic post came in that wasn't in the training data, the ML model would probably fail just as badly as the rule-based one. The rule-based model at least has explicit logic you can reason about — the ML model is a black box trained on too little data to trust for anything outside the training set.

## 5. Evaluation

**How you evaluated the model:**  
I ran all 18 posts through `predict_label` and compared each prediction to the corresponding entry in `TRUE_LABELS`. The model got 9 out of 18 correct — 50% accuracy.

**Examples of correct predictions:**

- `"I love this class so much"` → predicted `positive` ✓  
  The token `love` is in the positive word list and nothing contradicts it. Clean match.

- `"I am not happy about this"` → predicted `negative` ✓  
  This one only worked after I added negation handling. The token `happy` is positive, but `not` precedes it, so the score flips to -1. Without negation, this predicted `positive`, which was wrong.

- `"not bad, actually kind of fun"` → predicted `positive` ✓  
  `bad` gets negated by `not` (+1), and `fun` adds another +1. Final score is 2, which is the highest I saw in the dataset. The prediction is correct and the negation logic is doing real work here.

**Examples of incorrect predictions:**

- `"I absolutely love getting stuck in traffic"` → predicted `positive`, true label `negative`  
  The model sees `love` → +1 and nothing else it recognizes. It scores 1 and predicts positive. A human reads "getting stuck in traffic" as an obvious complaint, so the `love` is clearly sarcastic — but the model doesn't know what traffic means emotionally, and it doesn't know that `love` can be ironic.

- `"great, another Monday 🙄"` → predicted `positive`, true label `negative`  
  Similar to above. The model finds `great` → +1 and stops there. The 🙄 emoji is the clearest sarcasm signal in the whole sentence, but it's unknown to the model and contributes nothing. The final score is 1 → positive.

- `"the party was okay i guess... felt kinda invisible tbh"` → predicted `neutral`, true label `negative`  
  Score is 0 because none of the words appear in either word list. "Invisible", "kinda", "okay", "tbh" — all unknown. The model falls back to neutral by default. But a person reads "felt kinda invisible" as a clearly negative emotional experience. The model just doesn't have vocabulary for subtle or indirect language.

## 6. Limitations

**The model can't detect sarcasm.**  
This is the most consistent failure. Three posts in my dataset are sarcastic — `"I absolutely love getting stuck in traffic"`, `"I absolutely love when my alarm goes off at 5am 😐"`, and `"great, another Monday 🙄"` — and the model predicted `positive` for all three. In each case, it found a positive word (`love` or `great`), scored +1, and stopped. It has no way to know that those words are ironic because it doesn't consider what surrounds them.

**The model can never predict `mixed`.**  
Three posts in the dataset have the label `mixed`, and the model got all three wrong. The model's only output for a zero score is `neutral`, and there's no way to distinguish "balanced positive and negative signals" from "no signals at all" using a single number. To support `mixed`, the scoring logic would need to track hit counts separately and check whether both sides fired.

**The word lists are too small for real language.**  
Slang like "highkey", "lowkey", "no cap", and "hits different" score 0 every time because they aren't in the word lists. The same goes for most emojis. The model predicted `neutral` for `"no cap this hits different 🔥"` and `"highkey obsessed with this playlist rn 🎧❤️"` even though both clearly express strong positive feelings.

**The model is sensitive to exact word forms.**  
"Invisible", "kinda", "obsessed", "proud" — these all carry real sentiment, but the model scores them as 0. There's no stemming or fuzzy matching, so "stressed" and "stressing" are treated as completely different tokens even though they mean the same thing. If a word isn't in the list exactly as typed, it's invisible.

**The dataset is too small to trust the accuracy number.**  
50% on 18 examples doesn't mean much. The dataset was assembled by one person in a single session. It doesn't represent the full range of how people actually write, and there's no held-out test set — I'm evaluating on the same data I used to check my logic while building.

## 7. Ethical Considerations

**Misclassifying distress.**  
A message like "I'm not doing okay" could easily score neutral or even positive under this model (no matching words). If this kind of tool were deployed in a context where someone was trying to flag emotional distress — a wellness app, a moderation system — missing a negative signal could mean missing a person who actually needs help. The cost of a false neutral is not symmetric: it matters a lot more in some situations than others.

**Bias and scope.**  
The dataset I built reflects one person's casual internet English — specifically the kind of slang and shorthand common in Gen Z and millennial online spaces. Terms like "lowkey", "no cap", "highkey", and "bestie" all appear in the dataset, but they come from AAVE-influenced internet vernacular that not all English speakers use or write in. Someone expressing the same feelings in more formal English, British English, or a different dialect would likely score lower because their vocabulary doesn't overlap with the word lists I built.

The model is also tuned to short, punchy social media posts. It wasn't designed for full sentences, professional language, or anything longer than a tweet. A person writing "I find myself in a state of considerable distress" would probably score neutral because "distress" isn't in the word list and the model doesn't handle that register of language at all.

To put it plainly: this model is optimized for casual US internet English written by someone in their teens or twenties. Anyone writing differently — more formally, in a different dialect, or with cultural references outside that narrow window — is likely to be misread or ignored.

**Sarcasm misclassification.**  
The sarcasm failures aren't just accuracy problems — they're meaning reversals. When the model predicts `positive` for `"great, another Monday 🙄"`, it isn't just wrong by one label; it's reading the opposite of what the person meant. In a context where someone's emotional state is being inferred from their messages, that kind of error could produce a misleading picture of how they're actually doing.

**Privacy.**  
Analyzing mood from text means reading personal messages. Even if the analysis is automated, the system has access to how people express themselves privately. That's worth thinking about before applying this kind of model to real user data.

## 8. Ideas for Improvement

- Expand the word lists to include more slang and common emotional vocabulary — "awful", "dread", "hyped", "devastated", "proud" are all missing
- Map common emojis to sentiment signals — 😭 and 💀 carry strong meaning in context
- Track positive and negative hit counts separately so the model can output `mixed` when both sides fire
- Add more negation words — "no", "hardly", "barely", "without" can all flip meaning
- Expand the dataset significantly and split it into train and test sets so accuracy is measured on unseen examples
- Look into whether a simple ML model trained on more data would handle slang and sarcasm better — the rule-based model has hit its ceiling on those cases
