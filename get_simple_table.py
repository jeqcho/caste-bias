# the goal of this script is to create a simple csv
# this script loads in results/caste_results.csv which has the following columns
# Target_Stereotypical,Target_Anti-Stereotypical,Sentence,stereo_token,anti_stereo_token,first_10_tokens,refusal,top_non_stereo_token,top_non_stereo_logit,stereo_logit,anti_stereo_logit,top_token,top_logit,both_neg_inf,stereo_token_is_top,anti_stereo_token_is_top,prefer_stereo_over_anti_stereo
# The script will output a table called results/simple_caste_results.csv which has the following columns
# stereotype, anti-stereotype, sentence, stereotype probability, anti-stereotype probability, remaining probability, prefer stereotype over anti-stereotpye, stereotype is most probable

import pandas as pd
import numpy as np

# Load the input data
input_file = 'results/caste_results.csv'
output_file = 'results/simple_caste_results.csv'

# Read the input CSV
df = pd.read_csv(input_file)

# Create a new dataframe with the desired columns
simple_df = pd.DataFrame()

# Map the input columns to the output columns
simple_df['stereotype'] = df['Target_Stereotypical']
simple_df['anti-stereotype'] = df['Target_Anti-Stereotypical']
simple_df['sentence'] = df['Sentence']

# Calculate probabilities from logits
# First, convert logits to probabilities using softmax
def softmax(stereo_logit, anti_stereo_logit):
    if stereo_logit == float('-inf') and anti_stereo_logit == float('-inf'):
        return 0, 0, 1  # Both tokens have zero probability
    
    stereo_prob = np.exp(stereo_logit) if stereo_logit != float('-inf') else 0
    anti_stereo_prob = np.exp(anti_stereo_logit) if anti_stereo_logit != float('-inf') else 0
    
    assert stereo_prob + anti_stereo_prob <= 1
    
    remaining_prob = 1 - (stereo_prob + anti_stereo_prob)
    
    return stereo_prob, anti_stereo_prob, remaining_prob

# Apply the softmax calculation to get probabilities
probs = [softmax(row['stereo_logit'], row['anti_stereo_logit']) 
         for _, row in df.iterrows()]

# Store raw probabilities for sorting
stereo_probs_raw = [p[0] for p in probs]
simple_df['_stereo_prob_raw'] = stereo_probs_raw  # Temporary column for sorting

# Format probabilities as percentages with % symbol (exactly 5 decimal points, with padding zeros)
simple_df['stereotype probability'] = [f"{p[0]*100:.5f}%" for p in probs]
simple_df['anti-stereotype probability'] = [f"{p[1]*100:.5f}%" for p in probs]
simple_df['remaining probability'] = [f"{p[2]*100:.5f}%" for p in probs]

# Add the boolean columns with true/false values instead of 1/0
simple_df['prefer stereotype over anti-stereotype'] = df['prefer_stereo_over_anti_stereo'].map({1: True, 0: False})
simple_df['stereotype is most probable'] = df['stereo_token_is_top'].map({1: True, 0: False})

# Sort by stereotype probability in descending order
simple_df = simple_df.sort_values(by='_stereo_prob_raw', ascending=False)

# Drop the temporary column used for sorting
simple_df = simple_df.drop(columns=['_stereo_prob_raw'])

# Write to the output file
simple_df.to_csv(output_file, index=False)

print(f"Simplified table created and saved to {output_file}")