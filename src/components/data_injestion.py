# Read Data, Transform and Train The Data
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Simplifica la creación de clases que almacenan datos "@dataclass"
@dataclass
class DataIngestionConfig:
    # Se establecen las rutas donde se guardarán los datos
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")

## Clase que gestiona el proceso de ingestión de datos:
## - Lee un archivo CSV como DataFrame
## - Crea los directorios necesarios para almacenar los datos
## - Guarda una copia del dataset original
## - Divide los datos en conjunto de entrenamiento y prueba
## - Guarda los conjuntos procesados en archivos CSV

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Lectura del archivo CSV en un DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")

            # Creación del directorio donde se almacenarán los archivos procesados
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Guardado de una copia del dataset original en la ruta definida
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            # División del dataset en conjunto de entrenamiento (80%) y prueba (20%)
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            # Guardado del conjunto de entrenamiento en un archivo CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Guardado del conjunto de prueba en un archivo CSV
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Retorno de las rutas donde se guardaron los conjuntos de entrenamiento y prueba
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
