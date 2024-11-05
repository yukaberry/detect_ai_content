import os
import pandas as pd

def generate_dataset_sample(sample_size: int):
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)

    # Load huggingface dataset
    print("Load huggingface.co_human_ai_generated_text")
    huggingface_df = pd.read_csv(f'{module_dir_path}/../raw_data/huggingface.co_human_ai_generated_text/model_training_dataset.csv')
    huggingface_df['source']='huggingface.co_human_ai_generated_text'

    huggingface_human_text_df = huggingface_df[["human_text", "source"]]
    huggingface_human_text_df = huggingface_human_text_df.rename(columns={'human_text':'text'},)
    huggingface_human_text_df['generated'] = 0

    huggingface_ai_text_df = huggingface_df[["ai_text", "source"]]
    huggingface_ai_text_df = huggingface_ai_text_df.rename(columns={'ai_text':'text'},)
    huggingface_ai_text_df['generated'] = 1

    # Load kaggle dataset
    print("Load kaggle-ai-generated-vs-human-text")
    AI_Human_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/kaggle-ai-generated-vs-human-text/AI_Human.csv')
    AI_Human_enriched_df['source']='kaggle-ai-generated-vs-human-text'
    AI_Human_enriched_df = AI_Human_enriched_df[["text", "generated"]]

    # Load kaggle dataset
    print("daigt-v2-train-dataset")
    daigt_v2_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/daigt-v2-train-dataset/train_v2_drcat_02.csv')
    daigt_v2_enriched_df['source']='daigt-v2-train-dataset'
    daigt_v2_enriched_df = daigt_v2_enriched_df[["text"]]
    daigt_v2_enriched_df['generated'] = 1

    merged_df = pd.concat(objs=[
        huggingface_human_text_df,
        huggingface_ai_text_df,
        AI_Human_enriched_df,
        daigt_v2_enriched_df
        ])

    print(merged_df.shape)

    # Save it
    sample_df = merged_df.sample(sample_size)
    print(sample_df.shape)

    sample_path = f'{module_dir_path}/../raw_data/sample_dataset_{sample_size}.csv'
    sample_df.to_csv(sample_path, mode='w', index=True, header=True)
