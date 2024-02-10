import os
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from definitions import MAIN_DIR_PATH, NORMALIZED_DIR_PATH, HEATMAP_DIR_PATH

class DataPreparation:
    @staticmethod
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    @staticmethod
    def plot_heatmap_image(dataframe, output_path, folder_name):
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(dataframe.set_index('feature_name').T, cmap='Spectral_r', cbar=False, vmin=0, vmax=1000)
        ax.invert_yaxis()
        plt.axis('off')
        heatmap_filename = f"{folder_name}_heatmap.png"
        plt.savefig(os.path.join(output_path, heatmap_filename), bbox_inches='tight', pad_inches=0)
        plt.close()
        return os.path.join(output_path, heatmap_filename)

    @staticmethod
    def normalize_image(image_path, output_path):
        try:
            img = Image.open(image_path)
            img = img.convert('L')  # Convert to grayscale
            normalized_img_path = os.path.join(output_path, os.path.basename(image_path))
            img.save(normalized_img_path)
            return normalized_img_path
        except Exception as e:
            print(f"Error normalizing image {image_path}: {e}")
            return None

    @staticmethod
    def process_files(folder_path):
        files = sorted(os.listdir(folder_path), key=DataPreparation.extract_number)
        df = pd.DataFrame()

        for index, file in enumerate(files):
            if not file.endswith('.sp'):
                continue

            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.readlines()
                data_start = next((i for i, line in enumerate(content) if '#DATA' in line), -1) + 1
                if data_start == -1:
                    print(f"No '#DATA' found in file: {file}")
                    continue

                temp_df = pd.read_csv(file_path, skiprows=data_start, header=None, delim_whitespace=True, encoding='latin1')
                if df.empty:
                    df = temp_df
                    df.columns = ['feature_name', 'feature_value_0']
                else:
                    df[f'feature_value_{index}'] = temp_df.iloc[:, 1]

            except Exception as e:
                print(f"An error occurred while processing file {file}: {e}")

        return df

    @classmethod
    def process_data(cls):
        for root, dirs, files in os.walk(MAIN_DIR_PATH):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                df = cls.process_files(folder_path)
                if not df.empty:
                    heatmap_path = cls.plot_heatmap_image(df, HEATMAP_DIR_PATH, dir_name)
                    normalized_path = cls.normalize_image(heatmap_path, NORMALIZED_DIR_PATH)
                    print(f"Processed and normalized heatmap saved at: {normalized_path}")
