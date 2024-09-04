import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.covariance import EllipticEnvelope


def read_data_model(all_data, column_name):

    mask = all_data.target < 5
    #print(f"Y_data: {all_data.target.shape[0]}")

    x_data = all_data.data[mask]
    y_data = all_data.target[mask]
    #print(f"y_data: {y_data.shape[0]}")

    outlier_detector = EllipticEnvelope(contamination=0.05)  # Ajusta la contaminación según sea necesario
    is_inlier = outlier_detector.fit_predict(x_data)  # Identificar inliers (1) y outliers (-1)
    x_data = x_data[is_inlier == 1]
    y_data = y_data[is_inlier == 1]


    #print(f"y_data: {y_data.shape[0]}")

#    x_data = all_data.data
#    y_data = all_data.target

    # Split the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)

    # Split the training data into train_val and val
    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.30, random_state=42)

    #buscamos el indice de la columna de datos privados para añadir ruido
    global column_index
    column_index = list(all_data.feature_names).index(column_name)

    return x_train_val, x_val, x_test, y_train_val, y_val, y_test



def read_data_model_swap_output(all_data, column_name):
    x_train_val, x_val, x_test, y_train_val_orig, y_val_orig, y_test_orig = read_data_model(all_data, column_name)

    y_train_val = x_train_val[:, column_index].copy()
    y_val = x_val[:, column_index].copy()
    y_test = x_test[:, column_index].copy()

    x_train_val[:, column_index] = y_train_val_orig
    x_val[:, column_index] = y_val_orig
    x_test[:, column_index] = y_test_orig

    return x_train_val, x_val, x_test, y_train_val, y_val, y_test

# Function to add Laplace noise
def add_laplace_noise(data, epsilon):
      # Extraer los valores de la columna especificada
    column_values = data[:, column_index]
    
    # Calcular la sensibilidad (máximo - mínimo)
    min_value = column_values.min()
    max_value = column_values.max()
    sensitivity = max_value - min_value
    
    # Calcular la escala del ruido Laplaciano
    scale = sensitivity / epsilon
    
    # Generar ruido Laplaciano para esa columna
    noise = np.random.laplace(0, scale, size=column_values.shape)
    
    # Crear una copia de los datos originales
    data_noisy = data.copy()
    
    # Añadir el ruido solo a la columna especificada
    data_noisy[:, column_index] = column_values + noise

    
    return data_noisy


def add_gaussian_noise(data, epsilon):
    median_income_values = data[:, column_index]
    min_value = median_income_values.min()
    max_value = median_income_values.max()
    sensitivity = max_value - min_value
    delta = 1e-5

    sigma = sensitivity * np.sqrt(2 * np.log( 1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=data.shape)
    return data + noise


def rmse(data1, data2):
    mse_data = mse(data1, data2)
    return np.sqrt(mse_data)
