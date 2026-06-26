"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # Added posts — slang, emojis, sarcasm, mixed/ambiguous tone
    "lowkey been crying all day but at least the pizza was good 😅",
    "no cap this hits different 🔥",
    "I absolutely love when my alarm goes off at 5am 😐",
    "the party was okay i guess... felt kinda invisible tbh",
    "bestie said she's proud of me and now i'm crying happy tears 😭💕",
    "failed my exam but tbh im not even mad anymore 💀",
    "today was meh. not bad, not good, just meh 🤷",
    "highkey obsessed with this playlist rn 🎧❤️",
    # Sarcasm and negation examples — added while testing scoring edge cases
    "I absolutely love getting stuck in traffic",
    "not bad, actually kind of fun",
    "honestly don't know how I feel rn 😶",
    "great, another Monday 🙄",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # Labels for added posts
    "mixed",     # crying all day BUT pizza was good — genuinely split
    "positive",  # "no cap this hits different" — enthusiastic slang
    "negative",  # sarcasm: loving a 5am alarm is the opposite of what's said
    "negative",  # felt invisible — subtle low mood despite neutral opener
    "positive",  # happy tears from a compliment
    "mixed",     # bad outcome (failed exam) + resigned acceptance ("not even mad")
    "neutral",   # "meh" — explicitly neither positive nor negative
    "positive",  # "highkey obsessed" — strong positive slang
    # Sarcasm and negation labels
    "negative",  # sarcasm — "love" is ironic; true meaning is frustration
    "positive",  # negation flips "bad"; "fun" reinforces it
    "neutral",   # genuine uncertainty — no clear positive or negative signal
    "negative",  # sarcasm — "great" is ironic, eye-roll emoji confirms it
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
