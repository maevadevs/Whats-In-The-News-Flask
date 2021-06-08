# HELPER FUNCTIONS
# ****************

def get_summary(text, summary_tokenizer, summary_model):
    """
    Summarize a piece of text using T5 Pre-trained model
    """

    # Tokenize and tensorize the text
    # For tasks in T5, add the task verb. In our case: summarize
    # Max length of tokens supported by T5 is 512
    inputs = summary_tokenizer.encode(
        "summarize: ", 
        text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )

    # Generate the summaries: 18 words < summary < 150 words
    outputs = summary_model.generate(
        inputs, 
        max_length=150, 
        min_length=18, 
        length_penalty=5, 
        num_beams=2
    )

    # Convert summary output tensor IDs to text and return as summary
    return summary_tokenizer.decode(outputs[0])


def get_category_mapping(predicted_num):
    """
    Set of category-numbers mapping. 
    This is used to get back the actual category from the predicted_num.
    """

    # A predefined mapping of categories
    category_mapping = [
        ('Arts and Entertainment', 0),
        ('Automobiles', 1),
        ('Business', 2),
        ('Climate and Environment', 3),
        ('Energy', 4),
        ('Finance and Economics', 5),
        ('Food', 6),
        ('Global Healthcare', 7),
        ('Health and Wellness', 8),
        ('Legal and Crimes', 9),
        ('Life', 10),
        ('Markets and Investments', 11),
        ('Personal Finance', 12),
        ('Politics', 13),
        ('Real Estate', 14),
        ('Science and Technology', 15),
        ('Sports', 16),
        ('Travel and Transportation', 17),
        ('U.S.', 18),
        ('Wealth', 19),
        ('World', 20)
    ]

    # Now, get the actual category from the mapping and return
    for category, num in category_mapping:
        if num == predicted_num:
            return category
