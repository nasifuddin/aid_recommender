import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def load_processed_data(task_name: str = "train") -> pd.DataFrame:
    """
    Loads the combined processed X and y1/y2 data for EDA.
    """
    logger.info(f"Loading processed {task_name} data for EDA...")
    try:
        X = pd.read_csv(f'data/processed_X_{task_name}.csv')
        # --- FIX ---
        # Replaced deprecated 'squeeze=True' argument with .squeeze("columns")
        y1 = pd.read_csv(f'data/processed_y1_{task_name}.csv', header=None).squeeze("columns")
        y2 = pd.read_csv(f'data/processed_y2_{task_name}.csv', header=None).squeeze("columns")
        # --- END FIX ---

        # Combine for EDA
        df_eda = X.copy()
        df_eda['Participant_Selected_For_AID'] = y1.values
        df_eda['Aid_Type_Recomended'] = y2.values
        logger.info(f"EDA data loaded. Shape: {df_eda.shape}")
        return df_eda
    except FileNotFoundError as e:
        logger.error(f"Processed {task_name} data file not found for EDA: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {task_name} data for EDA: {e}")
        return None


def plot_target_distribution(df_eda: pd.DataFrame, target_col: str):
    """
    Plots the distribution of a target variable.
    """
    logger.info(f"Plotting distribution for target: {target_col}")
    if df_eda is None or target_col not in df_eda.columns:
        logger.error(f"Dataset or target column '{target_col}' not found.")
        return

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_eda, x=target_col)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/eda_target_{target_col}.png')
    plt.show()
    plt.close()


def plot_feature_correlation(df_eda: pd.DataFrame, target_col: str, top_n: int = 10):
    """
    Calculates correlation with the target and plots a heatmap for top N features.
    """
    logger.info(f"Calculating and plotting correlations for {target_col} (Top {top_n})")
    if df_eda is None:
        logger.error("Dataset not found.")
        return

    # Select only numeric columns for correlation
    numeric_df = df_eda.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        logger.error(f"Target column '{target_col}' not found in numeric columns for correlation.")
        return

    # Calculate correlation with target
    target_corr = numeric_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
    # Get top N features excluding the target itself
    top_features = target_corr[1:top_n+1].index.tolist() # Exclude target (index 0)
    logger.info(f"Top {top_n} features correlated with {target_col}: {top_features}")

    # Create a subset with top features and the target
    subset_df = numeric_df[top_features + [target_col]]
    corr_matrix = subset_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Heatmap (Top {top_n} Features for {target_col})')
    plt.tight_layout()
    plt.savefig(f'plots/eda_feature_correlation_{target_col}_top{top_n}.png')
    plt.show()
    plt.close()


def plot_feature_distribution_by_target(df_eda: pd.DataFrame, features_to_plot: list, target_col: str):
    """
    Plots distribution of selected features grouped by a target.
    """
    logger.info(f"Plotting feature distributions by {target_col} for features: {features_to_plot}")
    if df_eda is None:
        logger.error("Dataset not found.")
        return

    num_features = len(features_to_plot)
    if num_features == 0:
        logger.warning("No features provided for plotting.")
        return

    cols = 2
    rows = (num_features + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]  # Make it iterable if single subplot
    elif rows == 1 or cols == 1:
        axes = axes.flatten() # Flatten if single row or col
    else:
        axes = axes.flatten() # Flatten for multi-dim array

    for i, feature in enumerate(features_to_plot):
        if feature not in df_eda.columns:
            logger.warning(f"Feature '{feature}' not found in dataset.")
            continue
        sns.boxplot(data=df_eda, x=target_col, y=feature, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by {target_col}')
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel(feature)
        axes[i].tick_params(axis='x', rotation=45) # Rotate x-axis labels for target

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'plots/eda_feature_distributions_by_{target_col}.png')
    plt.show()
    plt.close()


def run_eda():
    """
    Main function to run the EDA pipeline.
    """
    logger.info("Starting EDA Pipeline...")

    df_train = load_processed_data("train")
    if df_train is not None:
        plot_target_distribution(df_train, 'Participant_Selected_For_AID')
        plot_target_distribution(df_train, 'Aid_Type_Recomended')

        plot_feature_correlation(df_train, 'Participant_Selected_For_AID', top_n=10)
        plot_feature_correlation(df_train, 'Aid_Type_Recomended', top_n=10)

        # Example features for distribution plot (adjust based on analysis or feature selection results)
        features_for_dist = ['income_pc', 'debt_to_income', 'dependency_ratio', 'productive_asset_index']
        plot_feature_distribution_by_target(df_train, features_for_dist, 'Participant_Selected_For_AID')
        plot_feature_distribution_by_target(df_train, features_for_dist, 'Aid_Type_Recomended')

    logger.info("EDA Pipeline completed. Plots saved to 'plots/' directory.")


# Example usage within the script (optional, can be called from notebook)
# if __name__ == "__main__":
#     run_eda()
