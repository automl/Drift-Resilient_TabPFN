import numpy as np
import pandas as pd
import torch
import random
import numbers
import os
from datetime import datetime

#### CONSTANTS ####

TASK_TYPE_MULTICLASS = "dist_shift_multiclass"
TASK_TYPE_REGRESSION = "dist_shift_regression"
MODULE_DIR = os.path.dirname(__file__)

#### LOADER FUNCTIONS ####


def get_datasets_dist_shift_multiclass_valid(include_synthetic=True):
    datasets = []
    if include_synthetic:
        ### Shifting Sine Wave
        datasets += [
            get_shifting_sin_data(
                num_domains=10,
                num_samples=150,
                random_state=1,
                step_size=0.5,
                min_distance=0.5,
                name="Sin Classification",
            )
        ]

        ### Rotated Blobs
        get_rotated_blobs_configs = [
            {
                "num_domains": 10,
                "num_samples_per_blob": 40,
                "num_blobs": 5,
                "rotation_sampler": lambda domain: domain * np.deg2rad(-20),
                "noise_standard_dev": 4.5,
                "radius": 25,
                "name": "Rotated Five Blobs - -20 deg",
                "random_state": 0,
                "center": (-5, 30),
            },
        ]

        for config in get_rotated_blobs_configs:
            datasets += [get_rotated_blobs(**config)]

        ### More deterministic blobs
        ## Deterministic movements

        # Moving Square
        square_centers = np.array([[0, 0], [0, 10], [10, 0], [10, 10]], dtype=float)
        square_movement_vectors = np.array(
            [[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=float
        )
        square_step_size = 2
        square_step_noise = np.zeros((4, 2), dtype=float)

        # Moving Diagonal Line
        diagonal_centers = np.array(
            [[0, 0], [np.cos(np.deg2rad(45)) * 20 + 7.5, np.sin(np.deg2rad(45)) * 20]],
            dtype=float,
        )
        diagonal_movement_vectors = np.array([[1, 1], [-1, -1]], dtype=float)
        diagonal_step_size = 4
        diagonal_step_noise = np.zeros((2, 2), dtype=float)

        get_blobs_configs = [
            # Moving Square
            {
                "num_domains": 6,
                "num_samples": 200,
                "num_features": 2,
                "num_classes": 4,
                "init_cluster_std": 1,
                "init_center_box": (-200, 200),
                "random_state": 100,
                "name": "Moving Blobs Dataset - 2D - 4 Classes - Moving Square",
                "centers": square_centers,
                "movement_vectors": square_movement_vectors,
                "step_size": square_step_size,
                "step_noise": square_step_noise,
            },
            # Moving Diagonal Line
            {
                "num_domains": 6,
                "num_samples": 200,
                "num_features": 2,
                "num_classes": 2,
                "init_cluster_std": 1,
                "init_center_box": (-200, 200),
                "random_state": 100,
                "name": "Moving Blobs Dataset - 2D - 2 Classes - Moving Diagonal Line",
                "centers": diagonal_centers,
                "movement_vectors": diagonal_movement_vectors,
                "step_size": diagonal_step_size,
                "step_noise": diagonal_step_noise,
            },
        ]

        for config in get_blobs_configs:
            datasets += [get_blobs(**config)]

    ### Real World Datasets
    datasets += [get_indian_liver_patients_data()]
    datasets += [get_istanbul_stock_exchange_data()]
    datasets += [get_diabetes_us_hospitals()]
    datasets += [get_airlines_data()]
    datasets += [get_diabetes_pima_indians_data()]
    datasets += [get_diabetes_questionaire_data()]
    datasets += [get_occupancy_detection_data()]
    datasets += [get_urban_traffic_sao_paulo_data()]

    return datasets


def get_datasets_dist_shift_multiclass_test(include_synthetic=True):
    datasets = []
    if include_synthetic:
        datasets += [get_rotated_moons_drain()]

        ### Prior probability shift
        get_binary_label_shift_configs = [
            # Two classes that fade in and out over time
            {
                "num_domains": 10,
                "num_samples": 200,
                "random_state": 1,
                "p_upper": 0.95,
                "p_lower": 0.05,
                "class_sep": 1e-15,
                "name": "Binary Label Shift Dataset - 2 Classes - 200 Samples - 10 Domains",
            }
        ]

        for config in get_binary_label_shift_configs:
            datasets += [get_binary_label_shift(**config)]

        ### Intersecting Blobs
        get_intersecting_blobs_config = [
            {
                "num_domains": 14,
                "num_samples": 40,
                "random_state": 0,
                "name": "Intersecting Blobs Dataset - 3 Classes - 120 Samples - 14 Domains",
            }
        ]

        for config in get_intersecting_blobs_config:
            datasets += [get_intersecting_blobs(**config)]

        datasets += [get_hyperplane_data()]

        datasets += [get_randomrbfdrift_data()]

        get_rotating_segments_configs = [
            {
                "num_domains": 10,
                "num_pieces": 4,
                "num_samples_per_domain": 150,
                "rotation_per_domain": 0.25,
                "max_radius": 0.2,
                "random_state": 0,
                "name": "Rotating Segments",
            }
        ]

        for config in get_rotating_segments_configs:
            datasets += [get_rotating_segments_data(**config)]
        get_sliding_circle_configs = [
            {
                "big_radius": 0.2,
                "small_radius": 0.075,
                "num_samples": 200,
                "num_domains": 10,
                "random_state": 0,
                "name": "Sliding Circle",
            }
        ]

        for config in get_sliding_circle_configs:
            datasets += [get_sliding_circle_data(**config)]

        get_shifting_two_spirals_configs = [
            {
                "num_points_per_arm": 100,
                "num_domains": 10,
                "twist": 1,
                "noise": 0.2,
                "random_state": 0,
                "name": "Shifting Two Spirals",
            }
        ]

        for config in get_shifting_two_spirals_configs:
            datasets += [get_shifting_two_spirals_data(**config)]

    ### Real World Datasets
    datasets += [get_free_light_chain_mortality_data()]
    datasets += [get_electricity_data()]
    datasets += [get_absenteeism_data()]
    datasets += [get_cleveland_heart_disease_data()]
    datasets += [get_parking_birmingham_data()]
    datasets += [get_housing_ames_data()]
    datasets += get_folktables_data(states=["MD"])  # Maryland
    datasets += [get_chess_data()]

    return datasets


#### HELPER FUNCTION ####


def dataframe_to_distribution_shift_ds(
    name, df, target, domain_name, task_type, dataset_source, shuffled=False
):
    from . import DistributionShiftDataset

    """
    Takes a dataframe and returns a DistributionShiftDataset of the given dataframe.

    This function preprocesses the given DataFrame by encoding categorical features, optionally shuffles
    the data within each domain, and then constructs a DistributionShiftDataset object.

    The function handles categorical data by encoding categories as integers. For multiclass tasks, the 
    target column is also treated as a categorical feature. 
    """
    if task_type == TASK_TYPE_MULTICLASS:
        df[target] = df[target].astype("category")

    cat_columns = df.select_dtypes(["object", "category", "bool"]).columns

    if len(cat_columns):
        df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # Shuffle within each domain
    if shuffled:
        df = (
            df.groupby(domain_name, observed=False)
            .apply(lambda x: x.sample(frac=1, random_state=42))
            .reset_index(drop=True)
        )

    domain = df[domain_name]
    y = df[target]

    # Drop the targets and the domain from the dataframe
    df = df.drop(columns=[target, domain_name])
    # Filter out the domain and the target from the categorical columns to be reported as categorical features
    # Beware that difference uses a set and thus the order is changed. This is not a problem however, as we
    # only check if the categories are in this set in the next step.
    cat_columns = cat_columns.difference([target, domain_name])

    # Get the indices in x that are categorical, this has to exclude the domain and the target.
    # Therefore it has to be done after the target and domain were dropped as otherwise the indices could be wrong.
    cat_columnns_indices = np.nonzero(df.columns.isin(cat_columns))[0].tolist()

    return DistributionShiftDataset(
        x=torch.tensor(df.values),
        y=torch.tensor(y.values).float()
        if isinstance(y, pd.Series)
        else torch.tensor(y).float(),
        task_type=task_type,
        dataset_source=dataset_source,
        name=name,
        attribute_names=list(df.columns),
        dist_shift_domain=torch.tensor(domain.values).int()
        if isinstance(domain, pd.Series)
        else torch.tensor(domain).int(),
        categorical_feats=cat_columnns_indices,
    )


#### DATASETS ####


def get_hyperplane_data():
    """
    Data generation based on scikit-multiflow HyperplaneGenerator

    @article{skmultiflow,
      author  = {Jacob Montiel and Jesse Read and Albert Bifet and Talel Abdessalem},
      title   = {Scikit-Multiflow: A Multi-output Streaming Framework },
      journal = {Journal of Machine Learning Research},
      year    = {2018},
      volume  = {19},
      number  = {72},
      pages   = {1-5},
      url     = {http://jmlr.org/papers/v19/18-251.html}
    }

    scikit-multiflow was released under BSD 3-Clause License
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/hyperplane_5feats_3drift_0-5mag_0-1noise.csv"),
        sep=",",
    )

    return dataframe_to_distribution_shift_ds(
        "Rotating Hyperplane",
        data,
        "Label",
        "Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_randomrbfdrift_data():
    """
    Data generation based on scikit-multiflow RandomRBFGeneratorDrift

    @article{skmultiflow,
      author  = {Jacob Montiel and Jesse Read and Albert Bifet and Talel Abdessalem},
      title   = {Scikit-Multiflow: A Multi-output Streaming Framework },
      journal = {Journal of Machine Learning Research},
      year    = {2018},
      volume  = {19},
      number  = {72},
      pages   = {1-5},
      url     = {http://jmlr.org/papers/v19/18-251.html}
    }

    scikit-multiflow was released under BSD 3-Clause License
    """
    data = pd.read_csv(
        os.path.join(
            MODULE_DIR,
            "data/randomrbfdrift_4classes_8features_10centroids_10shiftingcentroids_2changespeed.csv",
        ),
        sep=",",
    )

    return dataframe_to_distribution_shift_ds(
        "RandomRBF Drift",
        data,
        "Label",
        "Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_rotating_segments_data(
    num_pieces=4,
    num_domains=10,
    num_samples_per_domain=150,
    rotation_per_domain=0.25,
    max_radius=0.2,
    random_state=0,
    name="Rotating Segments",
):
    # The fixed angle width of each piece
    angle_width = 2 * np.pi / num_pieces

    # Create a local random generator
    rng = np.random.default_rng(random_state)

    # Generate uniformly distributed data points in the circle for all domains
    thetas = rng.uniform(0, 2 * np.pi, size=num_samples_per_domain)
    rs = np.sqrt(rng.uniform(0, max_radius**2, size=num_samples_per_domain))

    # Convert polar to cartesian coordinates for the base samples
    x_1s_base = rs * np.cos(thetas)
    x_2s_base = rs * np.sin(thetas)

    # Create an empty dataframe list
    df_list = []

    for domain in range(num_domains):
        # Calculate the piece each sample belongs to, and rotate it based on the domain
        pieces = ((thetas + domain * rotation_per_domain) // angle_width) % num_pieces

        # Assign labels based on the piece indices (alternating 0 and 1)
        labels = pieces.astype(int) % 2

        # Appending data to a temporary list
        domain_data = {
            "Feature1": x_1s_base,
            "Feature2": x_2s_base,
            "Domain": np.full(num_samples_per_domain, domain),
            "Label": labels,
        }
        df_list.append(pd.DataFrame(domain_data))

    # Concatenate dataframes from all domains
    df = pd.concat(df_list, ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_sliding_circle_data(
    big_radius=0.2,
    small_radius=0.075,
    num_samples=200,
    num_domains=10,
    random_state=0,
    name="Sliding Circle",
):
    """
    Generate the SlidingCircle dataset.

    big_radius: Radius of the bigger circle.
    small_radius: Radius of the sliding (smaller) circle.
    num_samples: Number of samples to fill in the larger circle.
    num_domains: Number of sliding circle positions (domains).
    """
    # Create a local random generator
    rng = np.random.default_rng(random_state)

    # Function to generate filled circle points
    def generate_filled_circle_points(radius, num_samples, center_x=0, center_y=0):
        theta = rng.uniform(0, 2 * np.pi, num_samples)
        r = np.sqrt(
            rng.uniform(0, radius**2, num_samples)
        )  # sqrt ensures uniform distribution
        x = r * np.cos(theta) + center_x
        y = r * np.sin(theta) + center_y
        return x, y

    # Define a function to determine if points are inside the sliding circle
    def inside_sliding_circle(x, y, circle_center):
        return (
            np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2)
            <= small_radius
        )

    # Generate samples for the larger circle, as this remains constant across domains
    x, y = generate_filled_circle_points(big_radius, num_samples)

    df = []
    # Slide the smaller circle around the inner perimeter of the bigger circle
    for domain in range(num_domains):
        # Calculate the center of the sliding circle for the current domain
        angle = np.pi / 2 - domain * (
            np.pi / (num_domains - 1)
        )  # To ensure pi rotation over the domains
        sliding_circle_center_x = (big_radius - 3 / 2 * small_radius) * np.cos(angle)
        sliding_circle_center_y = (big_radius - 3 / 2 * small_radius) * np.sin(angle)

        # Determine which points are inside the sliding circle
        inside_mask = inside_sliding_circle(
            x, y, [sliding_circle_center_x, sliding_circle_center_y]
        )

        # Class labels - 0 for the big circle and invert to 1 if inside the sliding circle
        labels = np.where(inside_mask, 1, 0)

        # Append samples to the dataframe
        domain_data = pd.DataFrame(
            {"Feature1": x, "Feature2": y, "Domain": domain, "Label": labels}
        )

        df += [domain_data]

    df = pd.concat(df, ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_shifting_two_spirals_data(
    num_points_per_arm=100,
    num_domains=10,
    twist=1,
    noise=0.2,
    random_state=0,
    name="Shifting Two Spirals",
):
    """
    Generate the two spirals dataset.
    """
    rng = np.random.RandomState(random_state)

    # Custom function to generate a single spiral arm
    def generate_spiral_arm(n, twist, noise):
        theta = np.linspace(0, 4 * np.pi, n)
        r = theta
        x = r * np.cos(theta * twist) + noise * rng.randn(n)
        y = r * np.sin(theta * twist) + noise * rng.randn(n)
        return x, y

    x_spiral1_base, y_spiral1_base = generate_spiral_arm(
        num_points_per_arm, twist, noise
    )
    x_spiral2_base, y_spiral2_base = (
        -x_spiral1_base,
        -y_spiral1_base,
    )  # The second spiral is a reflection of the first one

    # Calculate the normalized distances for the spirals
    dist_spiral1 = np.sqrt(x_spiral1_base**2 + y_spiral1_base**2)
    dist_spiral2 = np.sqrt(x_spiral2_base**2 + y_spiral2_base**2)

    max_dist = max(np.max(dist_spiral1), np.max(dist_spiral2))

    dist_spiral1_normalized = dist_spiral1 / max_dist
    dist_spiral2_normalized = dist_spiral2 / max_dist

    df = []
    for domain in range(num_domains):
        # Calculate the transition points for changing the class labels based on the current domain
        transition_point = domain / (num_domains - 1)

        labels_spiral1 = np.where(dist_spiral1_normalized <= transition_point, 1, 0)
        labels_spiral2 = np.where(dist_spiral2_normalized > 1 - transition_point, 0, 1)

        # Create the domain dataframe
        domain_data = pd.DataFrame(
            {
                "Feature1": np.concatenate([x_spiral1_base, x_spiral2_base]),
                "Feature2": np.concatenate([y_spiral1_base, y_spiral2_base]),
                "Domain": domain,
                "Label": np.concatenate([labels_spiral1, labels_spiral2]),
            }
        )

        df += [domain_data]

    df = pd.concat(df, ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_absenteeism_data():
    """

    @misc{misc_absenteeism_at_work_445,
      author       = {Martiniano,Andrea and Ferreira,Ricardo},
      title        = {{Absenteeism at work}},
      year         = {2018},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5X882}
    }

    https://archive.ics.uci.edu/dataset/445/absenteeism+at+work

    This dataset was licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/absenteeism_at_work.csv"), sep=";"
    )

    # Add a column to track the change in 'Month of absence'
    data["Season_Change"] = data["Seasons"].ne(data["Seasons"].shift()).astype(int)

    # Identify when the year changes based on 'Month of absence'
    data["Season_Domain"] = (data["Season_Change"] > 0).cumsum()

    # Remove the temporary columns
    data.drop(["Season_Change"], axis=1, inplace=True)

    data["Absenteeism time in hours"] = pd.qcut(
        data["Absenteeism time in hours"], 4, labels=False
    )

    # Remove the last three rows, seem to not fit the data description (time jump)
    data = data.iloc[:-3]

    # Cast columns to be categorical
    cat_columns = [
        "ID",
        "Reason for absence",
        "Day of the week",
        "Seasons",
        "Disciplinary failure",
        "Education",
        "Social drinker",
        "Social smoker",
        "Month of absence",
        "Season_Domain",
    ]

    data[cat_columns] = data[cat_columns].astype("category")

    return dataframe_to_distribution_shift_ds(
        "Absenteeism",
        data,
        "Absenteeism time in hours",
        "Season_Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_istanbul_stock_exchange_data():
    """
    @misc{misc_istanbul_stock_exchange_247,
      author       = {Akbilgic,Oguz},
      title        = {{ISTANBUL STOCK EXCHANGE}},
      year         = {2013},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C54P4J}
    }

    https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/data_istanbul_stock_exchange.csv"), sep=","
    )

    # Convert 'date' to datetime format
    data["date"] = pd.to_datetime(data["date"], format="%d-%b-%y")

    # Create 'Day' and 'Month' columns
    data["Day"] = data["date"].dt.day
    data["Month"] = data["date"].dt.month
    data["Year"] = data["date"].dt.year

    # Create a domain feature counting months since Jan 2009
    base_date = datetime(2009, 1, 1)
    data["Months_Since_Jan_2009"] = (data["date"].dt.year - base_date.year) * 12 + (
        data["date"].dt.month - base_date.month
    )

    # Remove the 'ISE_USD' and 'date' columns and predict the 'ISE_TL' column
    data.drop(["ISE_USD", "date"], axis=1, inplace=True)

    # Scale and round the 'ISE_TL' column
    data["ISE_TL"] = (data["ISE_TL"] * 100).round()

    # Clip values larger than 4 and smaller than -4
    data["ISE_TL"] = data["ISE_TL"].clip(-4, 4).astype(int)
    # Months_Since_Jan_2009
    return dataframe_to_distribution_shift_ds(
        "Istanbul Stock Exchange",
        data,
        "ISE_TL",
        "Months_Since_Jan_2009",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_preprocesed_diabetes():
    """
    @misc{misc_diabetes_130-us_hospitals_for_years_1999-2008_296,
      author       = {Clore,John, Cios,Krzysztof, DeShazo,Jon, and Strack,Beata},
      title        = {{Diabetes 130-US Hospitals for Years 1999-2008}},
      year         = {2014},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5230J}
    }

    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

    This dataset is licensed under the CC BY 4.0 license.

    - encounter_id can be seen as a time column
    - patient_nbr is the number of a patient, a patient can have multiple encounters
    """

    # from sklearn.impute import SimpleImputer

    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/diabetic_data.csv"),
        sep=",",
        na_values="?",
        low_memory=False,
    )

    # Remove the 'weight', 'payer_code', and 'medical_specialty' columns as they contain highly NaN values
    data.drop(["weight", "payer_code", "medical_specialty"], axis=1, inplace=True)

    # NaN for A1Cresult and max_glu_serum are not missing values, but rather a category
    data["A1Cresult"] = data["A1Cresult"].fillna("none")
    data["max_glu_serum"] = data["max_glu_serum"].fillna("none")

    # Drop remaining nan rows
    data.dropna(axis=0, inplace=True)

    missing_category_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]
    data[missing_category_columns] = data[missing_category_columns].astype("category")

    # Sort the dataset according to 'encounter_id'
    data.sort_values(by="encounter_id", ascending=True, inplace=True)

    # Keep only the first row for each unique 'patient_nbr'
    data.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)

    # Reset the index of the dataset
    data.reset_index(drop=True, inplace=True)

    data.drop(["patient_nbr", "encounter_id"], axis=1, inplace=True)

    return pd.DataFrame(data)


def get_diabetes_us_hospitals():
    """
    @misc{misc_diabetes_130-us_hospitals_for_years_1999-2008_296,
      author       = {Clore,John, Cios,Krzysztof, DeShazo,Jon, and Strack,Beata},
      title        = {{Diabetes 130-US Hospitals for Years 1999-2008}},
      year         = {2014},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5230J}
    }

    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = get_preprocesed_diabetes()

    age_mapping = {
        "[0-10)": 0,
        "[10-20)": 1,
        "[20-30)": 2,
        "[30-40)": 3,
        "[40-50)": 4,
        "[50-60)": 5,
        "[60-70)": 6,
        "[70-80)": 7,
        "[80-90)": 8,
        "[90-100)": 9,
    }

    data["age"] = data["age"].map(age_mapping)

    # Define the sample size for each group
    sample_size = 100

    # Perform stratified sampling
    stratified_data = (
        data.groupby("age")
        .apply(lambda x: x.sample(min(len(x), sample_size), random_state=1))
        .droplevel(0)
    )

    # Reset the index of the dataset
    stratified_data.reset_index(drop=True, inplace=True)

    stratified_data.sort_values(by="age", ascending=True, inplace=True)

    return dataframe_to_distribution_shift_ds(
        "Diabetes 130-US Hospitals",
        stratified_data,
        "readmitted",
        "age",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_urban_traffic_sao_paulo_data():
    """
    @misc{misc_behavior_of_the_urban_traffic_of_the_city_of_sao_paulo_in_brazil_483,
      author       = {Ferreira,Ricardo, Martiniano,Andrea, and Sassi,Renato},
      title        = {{Behavior of the urban traffic of the city of Sao Paulo in Brazil}},
      year         = {2018},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5902F}
    }

    https://archive.ics.uci.edu/dataset/483/behavior+of+the+urban+traffic+of+the+city+of+sao+paulo+in+brazil

    This dataset is licensed under the CC BY 4.0 license.
    """

    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/behaviour_of_urban_traffic_sao_paulo.csv"),
        sep=";",
    )

    data["Day"] = data["Hour (Coded)"].cumsum() // (27 * 28 / 2 + 0.0001)

    # Scale and round the 'Slowness in traffic (%)' column
    data["Slowness in traffic (%)"] = (
        (data["Slowness in traffic (%)"] // 7.5).round().clip(upper=2.0)
    )

    # Clip values larger than 10
    data["Slowness in traffic (%)"] = data["Slowness in traffic (%)"]

    return dataframe_to_distribution_shift_ds(
        "Urban Traffic Sao Paulo",
        data,
        "Slowness in traffic (%)",
        "Day",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_occupancy_detection_data():
    """
    @misc{misc_occupancy_detection__357,
      author       = {Candanedo,Luis},
      title        = {{Occupancy Detection }},
      year         = {2016},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5X01N}
    }

    https://archive.ics.uci.edu/dataset/357/occupancy+detection

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/occupancy_detection.csv"), sep=","
    )

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")
    data["Minute"] = data["date"].dt.minute
    data["Hour"] = data["date"].dt.hour
    data["Day"] = data["date"].dt.day

    data.drop(["Id", "date"], axis=1, inplace=True)

    # Randomly sample 1800 samples
    data_sample = data.sample(n=1800, random_state=1)

    # Sort the sampled data by the index to maintain the original order
    data_sample.sort_index(inplace=True)

    return dataframe_to_distribution_shift_ds(
        "Occupancy Detection",
        data_sample,
        "Occupancy",
        "Day",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_parking_birmingham_data():
    """
    @misc{misc_parking_birmingham_482,
      author       = {Stolfi,Daniel},
      title        = {{Parking Birmingham}},
      year         = {2019},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C51K5Z}
    }

    https://archive.ics.uci.edu/dataset/482/parking+birmingham

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(os.path.join(MODULE_DIR, "data/parking_birmingham.csv"), sep=",")

    # Convert 'LastUpdated' to datetime format
    data["LastUpdated"] = pd.to_datetime(
        data["LastUpdated"], format="%Y-%m-%d %H:%M:%S"
    )

    # Create 'Percentage_Occupied' column and discretize into intervals of 25 percent
    data["Percentage_Occupied"] = (data["Occupancy"] / data["Capacity"]) * 100
    # values smaller than 25 get 0, values between 25 and 50 get 1, values between 50 and 75 get 2, values larger than 75 get 3
    data["Percentage_Occupied"] = np.digitize(
        data["Percentage_Occupied"], bins=[25, 50, 75]
    ).astype(int)

    # Create 'Day', 'Week', and 'Domain' columns
    data["Hour"] = data["LastUpdated"].dt.hour
    data["Day"] = data["LastUpdated"].dt.day
    data["Month"] = data["LastUpdated"].dt.month
    data["Week_Dom"] = (
        data["LastUpdated"].dt.isocalendar().week
        - min(data["LastUpdated"].dt.isocalendar().week)
    ).astype(int)

    # Filter the data to only include the car park with the largest capacity
    data = data[data["SystemCodeNumber"] == "Others-CCCPS133"]

    # Remove 'LastUpdated', 'SystemCodeNumber', 'Week' and 'Year' columns
    data.drop(["LastUpdated", "SystemCodeNumber", "Occupancy"], axis=1, inplace=True)

    return dataframe_to_distribution_shift_ds(
        "Parking Birmingham",
        data,
        "Percentage_Occupied",
        "Week_Dom",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_airlines_data():
    """
    @misc{airlines,
      author       = {Bifet, Albert and Ikonomovska, Elena},
      title        = {{Data Expo competition}},
      year         = {2009},
      howpublished = {OpenML},
      url         = {https://www.openml.org/search?type=data&sort=runs&id=1169&status=active}
    }

    https://openml.org/search?type=data&sort=runs&status=active&id=1169
    https://github.com/datasets/openml-datasets/blob/master/data/airlines/datapackage.json

    https://www.kaggle.com/datasets/ulrikthygepedersen/airlines-delay/discussion

    This dataset is licensed under the Open Data Commons Public Domain Dedication and License (ODC-PDDL).
    """
    df = pd.read_csv(os.path.join(MODULE_DIR, "data/airlines.csv"))

    cat_indices = ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek"]
    df[cat_indices] = df[cat_indices].apply(lambda x: x.astype("category"))

    df["Time"] = df["Time"] // 60

    df.sort_values(by="Time", inplace=True)

    # Perform stratified sampling
    samples_per_interval = (
        60  # calculate how many samples should be taken from each interval
    )
    stratified_df = (
        df.groupby("Time")
        .apply(lambda x: x.sample(min(samples_per_interval, len(x)), random_state=1))
        .droplevel(0)
    )

    return dataframe_to_distribution_shift_ds(
        "Airlines",
        stratified_df,
        target="target",
        domain_name="Time",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_electricity_data():
    """..
    @Book{ harries1999splice,
        author = { Harries, Michael},
        title = { Splice-2 comparative evaluation: Electricity Pricing},
        publisher = { University of New South Wales, School of Computer Science and Engineering [Sydney] },
        year = { 1999 },
        type = { Book, Online },
        url = { http://nla.gov.au/nla.arc-32869 },
        language = { English },
        subjects = { Machine learning },
        life-dates = { 1999 -  },
        catalogue-url = { https://nla.gov.au/nla.cat-vn3513275 },
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/elec2.csv"),
        sep=",",
        na_values="?",
        skipinitialspace=True,
    )

    # Convert the 'date' column to string type
    df["date"] = df["date"].astype(str)

    # Extract the year, month, and day from the 'date' column
    df["date"] = (
        "19"
        + df["date"].str[:2]
        + "-"
        + df["date"].str[2:4]
        + "-"
        + df["date"].str[4:6]
    )

    # Date ranged from 7 May 1996 to 5 December 1998
    # Create a new 'date' column using the extracted year, month, and day
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Drop nan rows
    df.dropna(inplace=True)

    # Drop half of the rows to reduce the size of the dataset
    df = df[df["half_hour_interval"] % 4 == 0]

    # Create the domain column in which every two week period is considered a new domain
    # Create a grouper object based on 'date' and group by it
    grouper = pd.Grouper(key="date", freq="1W")

    # Create a new column 'domain', each unique group will have a unique identifier
    df["domain"] = df.groupby(grouper).ngroup()

    # Drop the first and last group, since those can be incomplete weeks
    df = df[df["domain"] != 0]
    df = df[df["domain"] != df["domain"].max()]

    # Change the data type of some columns
    df = df.astype(
        {
            "day_of_week": "category",
            "half_hour_interval": "int",
            "nsw_demand": "float",
            "v_demand": "float",
            "transfer": "float",
        }
    )

    # Drop the columns not intended for the model
    df.drop(["date", "nsw_prize", "v_prize"], axis=1, inplace=True)

    # Subsample a range of 15 domains to reduce the size of the dataset even further
    np.random.seed(0)  # Fixing the seed for reproducibility

    range = 15
    max_start = df["domain"].max() - range
    start = np.random.randint(1, max_start + 1)
    end = start + range

    df = df[(df["domain"] >= start) & (df["domain"] < end)]

    return dataframe_to_distribution_shift_ds(
        name="Electricity",
        df=df,
        target="target",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_chess_data():
    """
    @article{vzliobaite2011combining,
      title={Combining similarity in time and space for training set formation under concept drift},
      author={{\v{Z}}liobait{\.e}, Indr{\.e}},
      journal={Intelligent Data Analysis},
      volume={15},
      number={4},
      pages={589--611},
      year={2011},
      publisher={IOS Press}
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/chess.csv"), sep=",", skipinitialspace=True
    )

    # Date ranged from 7 December 2007 to 26 March 2010
    # Build the date column out of the year, month and day column
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Sort by the date
    df = df.sort_values(by="date").reset_index(drop=True)

    # Group every 20 consecutive games as a single domain, this should better track the progress a player makes
    # as the time contains gaps, that are probably not relevant to a player's progress
    df["domain"] = df.index // 20

    # Change the data type of some columns
    cat_columns = ["white/black", "type", "outcome"]
    df[cat_columns] = df[cat_columns].astype("category")

    # Drop the columns not intended for the model
    df.drop(["date"], axis=1, inplace=True)

    return dataframe_to_distribution_shift_ds(
        name="Chess",
        df=df,
        target="outcome",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_housing_ames_data():
    """
    @article{cock2011ames,
        author = {De Cock, Dean},
        year = {2011},
        month = {11},
        pages = {},
        title = {Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project},
        volume = {19},
        journal = {Journal of Statistics Education},
        doi = {10.1080/10691898.2011.11889627}
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/housing_ames.csv"), keep_default_na=False
    )

    orderings = {
        "Street": ["Pave", "Grvl"],
        "Alley": ["Pave", "Grvl", "NA"],
        "Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO"],
        "LandSlope": ["Gtl", "Mod", "Sev"],
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"],
        "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"],
        "CentralAir": ["Y", "N"],
        "Electrical": ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix", "NA"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "Functional": ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
        "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "GarageFinish": ["Fin", "RFn", "Unf", "NA"],
        "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "PavedDrive": ["Y", "P", "N"],
        "PoolQC": ["Ex", "Gd", "TA", "Fa", "NA"],
        "Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"],
    }

    for col, ordering in orderings.items():
        df[col] = pd.Categorical(df[col], categories=ordering, ordered=True)

    # Set MSSubClass as categorical since it denotes a type not a number
    df["MSSubClass"] = pd.Categorical(df["MSSubClass"])

    # Delete the ID column
    df.drop("Id", axis=1, inplace=True)

    # Create the domain column in which every 10 year period is considered a new domain
    min_year = df["YearBuilt"].min()
    max_year = df["YearBuilt"].max()

    df.sort_values(by=["YearBuilt"], inplace=True)

    # Create a new column 'domain', each unique group will have a unique identifier
    # Define bins as every 15 years
    bins_domain = np.arange(min_year, max_year + 15, 15)

    df["domain"] = pd.cut(df["YearBuilt"], bins=bins_domain, right=False).cat.codes

    bins_target = [0, 125000, 300000, np.inf]
    labels = ["<= 125k", "125k-300k", "> 300k"]

    # Discretize the sale price of a housing, use fixed size bins to create 5 bins
    df["SalePrice"] = pd.cut(
        df["SalePrice"], bins=bins_target, labels=labels, include_lowest=True
    ).cat.codes

    return dataframe_to_distribution_shift_ds(
        name="Ames Housing Prices",
        df=df,
        target="SalePrice",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_folktables_data(states):
    from folktables import (
        ACSDataSource,
        ACSIncome,
        ACSPublicCoverage,
        ACSMobility,
        ACSEmployment,
        ACSTravelTime,
    )

    # Create a dictionary to store all dataframes
    all_data = {
        "ACSIncome": None,
        "ACSPublicCoverage": None,
        #'ACSMobility': None,
        "ACSEmployment": None,
        #'ACSTravelTime': None
    }

    acs_data_2015_2019 = ACSDataSource(
        survey_year=2019,
        horizon="5-Year",
        survey="person",
        root_dir="./datasets/data/folktables",
    ).get_data(states=states, download=True)
    acs_data_2017_2021 = ACSDataSource(
        survey_year=2021,
        horizon="5-Year",
        survey="person",
        root_dir="./datasets/data/folktables",
    ).get_data(states=states, download=True)

    # Features that are in the 2015-2019 dataset but not in the 2017-2021 dataset
    # {'MLPI', 'MLPK'}
    # Features that are in the 2017-2021 dataset but not in the 2015-2019 dataset
    # {'MLPIK'}
    # Since our tasks dont use them, lets just drop them

    acs_data_2015_2019.drop(columns=["MLPI", "MLPK"], inplace=True)
    acs_data_2017_2021.drop(columns=["MLPIK"], inplace=True)

    # Build a year column that is the year an individual was surveyed
    acs_data_2015_2019["YEAR"] = (
        acs_data_2015_2019["SERIALNO"].apply(lambda x: x[:4]).astype(int)
    )
    acs_data_2017_2021["YEAR"] = (
        acs_data_2017_2021["SERIALNO"].apply(lambda x: x[:4]).astype(int)
    )

    # Merge the two datasets
    acs_data = pd.concat(
        [acs_data_2015_2019, acs_data_2017_2021[acs_data_2017_2021["YEAR"] > 2019]]
    )

    # Rename the columns that were renamed in 2019 by ACS to be consistent with the feature names as stated by folktables
    acs_data.rename(columns={"RELSHIPP": "RELP", "JWTRNS": "JWTR"}, inplace=True)

    used_cat_columns = {
        "DREM",
        "FER",
        "POWPUMA",
        "SCHL",
        "MIL",
        "MAR",
        "CIT",
        "TARGET",
        "DIS",
        "PUMA",
        "ANC",
        "NATIVITY",
        "ESP",
        "COW",
        "RELP",
        "RAC1P",
        "OCCP",
        "POBP",
        "DEAR",
        "ESR",
        "ST",
        "DEYE",
        "MIG",
        "GCL",
        "JWTR",
        "SEX",
    }

    # Data processing for each task
    tasks = [
        (ACSIncome, "ACSIncome"),
        (ACSPublicCoverage, "ACSPublicCoverage"),
        # (ACSMobility, 'ACSMobility'),
        (ACSEmployment, "ACSEmployment"),
    ]
    # (ACSTravelTime, 'ACSTravelTime')]

    random_state = 0
    for task, task_name in tasks:
        # We'd like to keep the new YEAR feature in our tasks
        if "YEAR" not in task.features:
            task.features.append("YEAR")

        features, labels, _ = task.df_to_pandas(acs_data)
        features["TARGET"] = labels

        # Convert the categorical features to be categorical and not numerical
        cat_columns = list(set.intersection(set(features.columns), used_cat_columns))
        features[cat_columns] = features[cat_columns].apply(
            lambda x: x.astype("category")
        )

        instances = 1300
        instances_per_year = round(instances / 7)  # 7 years of data

        # Subsample the data to make it more manageable
        subsampled_dfs = []
        for year in range(2015, 2021 + 1):
            domain_instances = features[features["YEAR"] == year].shape[0]
            # subsampled_instances = round(expected_instances_per_year * domain_instances / overall_instances)

            # Use stratified sampling to ensure that the distribution of the target is preserved
            subsampled_df = (
                features[features["YEAR"] == year]
                .groupby("TARGET", observed=False)
                .apply(
                    lambda x: x.sample(
                        frac=instances_per_year / domain_instances,
                        random_state=random_state,
                    )
                )
                .droplevel(0)
            )

            # Restore the order relative to each other
            subsampled_df.sort_index(inplace=True)

            subsampled_dfs.append(subsampled_df)

            random_state += 1

        # concatenate all subsampled DataFrames
        features = pd.concat(subsampled_dfs)

        all_data[task_name] = features

    # Reset index of each DataFrame
    for df in all_data.values():
        df.reset_index(drop=True, inplace=True)

    dataset_list = []

    for task, task_name in tasks:
        dataset = dataframe_to_distribution_shift_ds(
            name=f"Folktables - {task_name} - {' | '.join(states)}",
            df=all_data[task_name],
            target="TARGET",
            domain_name="YEAR",
            task_type=TASK_TYPE_MULTICLASS,
            dataset_source="real-world",
            shuffled=False,
        )
        dataset_list.append(dataset)

    return dataset_list


def get_diabetes_questionaire_data():
    """
    @inbook{islam2010likelihood,
        author = {Islam, M M Faniqul and Ferdousi, Rahatara and Rahman, Sadikur and Bushra, Humayra},
        year = {2020},
        month = {01},
        pages = {113-125},
        title = {Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques},
        isbn = {978-981-13-8797-5},
        doi = {10.1007/978-981-13-8798-2_12}
    }

    https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/diabetes_questionaire.csv"), sep=","
    )

    data.sort_values("Age", inplace=True)

    # Build the domain, which is an age category of 5 year intervals
    data["domain"] = pd.cut(
        data["Age"],
        bins=np.arange(
            data["Age"].min() - data["Age"].min() % 5, data["Age"].max() + 6, 5
        ),
        right=False,
    ).cat.codes

    # Convert the class to binary labels
    data["class"] = data["class"].replace({"Positive": 1, "Negative": 0})
    data["class"] = (
        data["class"].replace({"Positive": 1, "Negative": 0}).infer_objects(copy=False)
    )

    return dataframe_to_distribution_shift_ds(
        "Diabetes Questionaire",
        data,
        "class",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_indian_liver_patients_data():
    """
    @misc{ramana2012ilpd,
        author       = {Ramana,Bendi and Venkateswarlu,N.},
        title        = {{ILPD (Indian Liver Patient Dataset)}},
        year         = {2012},
        howpublished = {UCI Machine Learning Repository},
        note         = {{DOI}: https://doi.org/10.24432/C5D02C}
    }

    https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/indian_liver_patient.csv"), sep=","
    )

    data.sort_values("Age", inplace=True)

    # Rename the target column
    data.rename(columns={"Dataset": "Target"}, inplace=True)

    # Map the target to 0 and 1
    # 0 = Liver patient / 1 = no liver patient
    data["Target"] -= 1

    # Classify the gender to be categorical
    data["Gender"] = data["Gender"].astype("category")

    # Build the domain, which is an age category of 5 year intervals
    data["domain"] = pd.cut(
        data["Age"],
        bins=np.arange(
            data["Age"].min() - data["Age"].min() % 5, data["Age"].max() + 6, 5
        ),
        right=False,
    ).cat.codes

    return dataframe_to_distribution_shift_ds(
        "Indian Liver Patients",
        data,
        "Target",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_diabetes_pima_indians_data():
    """
    @article{smith1988using,
        author = {Smith, Jack and Everhart, J. and Dickson, W. and Knowler, W. and Johannes, Richard},
        year = {1988},
        month = {11},
        pages = {},
        title = {Using the ADAP Learning Algorithm to Forcast the Onset of Diabetes Mellitus},
        volume = {10},
        journal = {Proceedings - Annual Symposium on Computer Applications in Medical Care}
    }

    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/diabetes_pima_indians.csv"), sep=","
    )

    data.sort_values("Age", inplace=True)

    # Build the domain, which is an age category of 5 year intervals
    data["domain"] = pd.cut(
        data["Age"],
        bins=np.arange(
            data["Age"].min() - data["Age"].min() % 2, data["Age"].max() + 3, 2
        ),
        right=False,
    ).cat.codes
    return dataframe_to_distribution_shift_ds(
        "Pima Indians Diabetes",
        data,
        "Outcome",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_cleveland_heart_disease_data():
    """
    @misc{misc_heart_disease_45,
        author       = {Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert},
        title        = {{Heart Disease}},
        year         = {1988},
        howpublished = {UCI Machine Learning Repository},
        note         = {{DOI}: https://doi.org/10.24432/C52P4X}
    }

    https://archive.ics.uci.edu/dataset/45/heart+disease
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/cleveland_heart_disease.csv"),
        sep=",",
        na_values="?",
    )

    # Drop the few nan rows
    data.dropna(inplace=True)

    # Cast all columns but oldpeak to be integer
    cols = data.columns.drop("oldpeak")
    data[cols] = data[cols].apply(pd.to_numeric, downcast="integer", errors="coerce")

    # Set the type of some columns to be categorical
    cat_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    data[cat_columns] = data[cat_columns].apply(lambda x: x.astype("category"))

    data.sort_values("age", inplace=True)

    # Build the domain, which is an age category of 5 year intervals
    data["domain"] = pd.cut(
        data["age"],
        bins=np.arange(
            data["age"].min() - data["age"].min() % 4, data["age"].max() + 5, 4
        ),
        right=False,
    ).cat.codes

    # treat all deseases as one class
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

    return dataframe_to_distribution_shift_ds(
        "Cleveland Heart Disease",
        data,
        "target",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_free_light_chain_mortality_data():
    """
    @article{dispenzieri2012nonclonal,
        author = {Dispenzieri, Angela and Katzmann, Jerry and Kyle, Robert and Larson, Dirk and Therneau, Terry and Colby, Colin and Clark, Raynell and Mead, Graham and Kumar, Shaji and Melton, L and Rajkumar, S},
        year = {2012},
        month = {06},
        pages = {517-23},
        title = {Use of Nonclonal Serum Immunoglobulin Free Light Chains to Predict Overall Survival in the General Population},
        volume = {87},
        journal = {Mayo Clinic proceedings. Mayo Clinic},
        doi = {10.1016/j.mayocp.2012.03.009}
    }

    https://www.kaggle.com/datasets/nalkrolu/assay-of-serum-free-light-chain
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/free_light_chain_mortality.csv"), sep=","
    )

    # Drop the chapter since it leaks information whether the patient died or not,
    # which is the target variable
    data.drop(["chapter"], axis=1, inplace=True)

    # Drop the first column since it is just an index
    data.drop(["Unnamed: 0"], axis=1, inplace=True)

    # Convert categorical variables
    cat_columns = ["death", "mgus", "flc.grp", "sex"]
    data[cat_columns] = data[cat_columns].astype("category")

    # Drop rows that are nan since we are subsampling the data anyway
    data.dropna(axis=0, inplace=True)

    subsampled_dfs = []
    for dom in data["sample.yr"].unique():
        domain_instances = data[data["sample.yr"] == dom].shape[0]

        # Use stratified sampling to ensure that the distribution of the target is preserved
        subsampled_df = (
            data[data["sample.yr"] == dom]
            .groupby("death", observed=False)
            .apply(
                lambda x: x.reset_index().sample(
                    frac=min(1.0, 80 / domain_instances), random_state=42
                )
            )
            .droplevel(0)
        )

        # Restore the order relative to each other
        subsampled_df.sort_index(inplace=True)

        subsampled_dfs.append(subsampled_df)

    # concatenate all subsampled DataFrames
    data = pd.concat(subsampled_dfs)

    # Restore the order relative to each other
    data.sort_values("sample.yr", inplace=True)

    return dataframe_to_distribution_shift_ds(
        "Free Light Chain Mortality",
        data,
        "death",
        "sample.yr",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_rotated_moons(
    num_domains,
    num_samples,
    rotation_sampler,
    noise_standard_dev,
    name="Rotated Two Moons Dataset",
    random_state=0,
):
    """
    Generates the 2-Moons dataset according to the given specification and returns it as a DistributionShiftDataset
    """
    from sklearn.datasets import make_moons

    if random_state is None:
        random_state = random.randint(0, 100000000)

    def rotate(X, y, phi, centroids=((0, 0), (0, 0))):
        transl_x1 = np.array([centroids[0][0], centroids[0][1]])
        transl_x2 = np.array([centroids[1][0], centroids[1][1]])
        rotation = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

        X[y == 0] = (X[y == 0] - transl_x1) @ rotation.T + transl_x1
        X[y == 1] = (X[y == 1] - transl_x2) @ rotation.T + transl_x2

    df = pd.DataFrame(columns=["Feature1", "Feature2", "Label", "Domain"])
    for i in range(num_domains):
        X, y = make_moons(
            n_samples=num_samples,
            shuffle=True,
            noise=noise_standard_dev,
            random_state=random_state,
        )
        rotate(X, y, rotation_sampler(i))
        domain_df = pd.DataFrame(data=X, columns=["Feature1", "Feature2"])
        domain_df["Label"] = y
        domain_df["Domain"] = i
        df = pd.concat([df, domain_df], ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_rotated_blobs(
    num_domains,
    num_samples_per_blob,
    num_blobs,
    rotation_sampler,
    noise_standard_dev,
    radius,
    center=(0, 0),
    name="Rotated Blobs Dataset",
    random_state=0,
):
    """
    Function to generate rotating blobs dataset.

    Parameters:
    - num_domains: Number of different rotation angles (or domains) for the blobs.
    - num_samples_per_blob: Number of samples per blob per domain.
    - num_blobs: Number of blobs to generate.
    - rotation_sampler: Function to determine the rotation angle based on the domain index.
    - noise_standard_dev: Standard deviation of Gaussian noise added to the data.
    - radius: Distance of the blobs' centers from the center of rotation.
    - center: The center of rotation.
    - name: Name of the dataset.
    - random_state: Seed for the random number generator.

    Returns:
    - DistributionShiftDataset object containing the generated data.
    """
    from sklearn.datasets import make_blobs

    # If no random state is provided, generate a random seed.
    if random_state is None:
        random_state = np.random.randint(0, 100000000)

    # Define the initial positions of the blobs.
    blob_centers = [
        (
            center[0] + radius * np.cos(2 * np.pi * i / num_blobs),
            center[1] + radius * np.sin(2 * np.pi * i / num_blobs),
        )
        for i in range(num_blobs)
    ]

    def rotate(X, y, phi, center):
        """
        Function to rotate points around a specified center.

        Parameters:
        - X: Coordinates of the points.
        - y: Labels of the points.
        - phi: Rotation angle.
        - center: The center of rotation.

        Returns: None. The points X are modified in place.
        """
        # Define the rotation matrix.
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        # Shift the points so that the center of rotation is at the origin,
        # rotate the points, and then shift the points back.
        X[:] = (X - center) @ rotation.T + center

    # Initialize the DataFrame to store the data.
    df = []
    for i in range(num_domains):
        # Generate the blobs, change the random state of each blob in each domain.
        X, y = make_blobs(
            n_samples=num_samples_per_blob * num_blobs,
            centers=blob_centers,
            cluster_std=noise_standard_dev,
            random_state=random_state + i,
        )
        # Rotate the blobs.
        rotate(X, y, rotation_sampler(i), center)
        # Create a DataFrame from the generated data and append it to the main DataFrame.
        domain_df = pd.DataFrame(data=X, columns=["Feature1", "Feature2"])
        domain_df["Label"] = y
        domain_df["Domain"] = i

        df += [domain_df]

    df = pd.concat(df, ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_blobs(
    num_domains,
    num_samples,
    num_features=2,
    num_classes=2,
    init_cluster_std=1.0,
    init_center_box=(-10.0, 10.0),
    random_state=None,
    name="Moving Blobs Dataset",
    centers=None,
    movement_vectors=None,
    step_size=None,
    step_noise=None,
):
    """
    Generates the sklearn blobs dataset according to the given specification and returns it as a DistributionShiftDataset
    """
    from sklearn.datasets import make_blobs

    if random_state is None:
        random_state = random.randint(0, 10000000)

    np.random.seed(random_state)

    if centers is not None:
        assert centers.shape == (
            num_classes,
            num_features,
        ), "The centers must be a matrix of shape (num_classes, num_features)"
        centers = centers
    else:  # In case no centers are given, we will generate them randomly
        centers = num_classes

    # Generate the first domain and return the sampled centers that we will move in successive domains
    X, y, centers = make_blobs(
        n_samples=num_samples,
        n_features=num_features,
        centers=centers,
        cluster_std=init_cluster_std,
        center_box=init_center_box,
        shuffle=True,
        random_state=random_state - 1,
        return_centers=True,
    )

    if movement_vectors is not None:
        assert movement_vectors.shape == (
            num_classes,
            num_features,
        ), "The movement vectors must be a matrix of shape (num_classes, num_features)"
    else:  # Otherwise, sample random vectors for each class
        movement_vectors = np.random.uniform(-1, 1, size=centers.shape)

    unit_vectors = movement_vectors / np.linalg.norm(
        movement_vectors, axis=1, keepdims=True
    )

    df = []
    for i in range(num_domains):
        domain_df = pd.DataFrame(
            data=X, columns=[f"Feature{i + 1}" for i in range(num_features)]
        )
        domain_df["Label"] = y
        domain_df["Domain"] = i

        df += [domain_df]

        if step_size is not None:
            assert isinstance(
                step_size, numbers.Number
            ), "The step size must be a number"
        else:  # Otherwise, move a random amount in the direction of the random vector and add some noise to the path
            step_size = np.random.normal(2, 1)

        if step_noise is not None:
            assert (
                step_noise.shape == centers.shape
            ), "The step noise must be a matrix of shape (num_classes, num_features)"
        else:  # Otherwise, add noise to each direction of each class
            step_noise = np.random.normal(0, 1.5, size=centers.shape)

        centers += step_size * unit_vectors + step_noise

        # Generate the next domain
        X, y = make_blobs(
            n_samples=num_samples,
            n_features=num_features,
            centers=centers,
            cluster_std=init_cluster_std,
            center_box=init_center_box,
            shuffle=True,
            random_state=random_state - 1,
        )

    df = pd.concat(df, ignore_index=True)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_intersecting_blobs(
    num_domains=14,
    num_samples=40,
    random_state=0,
    name="Intersecting Blobs Dataset",
):
    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Initialize the centers and standard deviations of the blobs
    centers = np.array([[-16, 0.0], [6.0, 12.0], [4.0, -8.0]])
    std_devs = np.array([1.1, 0.9, 1.0])

    xs = []
    ys = []
    domains = []

    # For each domain
    for i in range(num_domains):
        # Move the blobs
        centers += (
            np.random.normal(
                loc=[2.0, 0.2, -1, -2.5, -0.1, 0.5], scale=0.1, size=6
            ).reshape(3, 2)
            * 0.9
        )

        # Generate data points for each blob
        for idx, (center, std_dev) in enumerate(zip(centers, std_devs)):
            points = np.random.normal(loc=center, scale=std_dev, size=(num_samples, 2))
            xs.append(points)
            ys.extend([idx] * num_samples)
            domains.extend([i] * num_samples)

        # Vary the standard deviation of the blobs
        std_devs += np.random.normal(loc=0.0, scale=0.1, size=3)
        std_devs = np.clip(
            std_devs, 0.1, np.inf
        )  # Ensure that the standard deviations remain positive

    # Concatenate the datasets
    x = np.vstack(xs)
    y = np.array(ys)
    domain = np.array(domains)

    features = [f"Feature{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=features)
    df["Label"] = y
    df["Domain"] = domain

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_binary_label_shift(
    num_domains=10,
    num_samples=200,
    random_state=1,
    p_upper=0.9,
    p_lower=0.1,
    name="Binary Label Shift Dataset",
    class_sep=2.0,
):
    from sklearn.datasets import make_classification
    import pandas as pd

    Xs = []
    ys = []
    for i in range(num_domains):
        p = p_lower + (p_upper - p_lower) * (1 - i / (num_domains - 1))

        X, y = make_classification(
            n_samples=num_samples,
            n_classes=2,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=2,
            weights=[p, 1 - p],
            flip_y=0.00,
            random_state=random_state,
            scale=1,
            shift=0,
            class_sep=class_sep,
        )

        Xs += [X]
        ys += [y]

    # Concatenate the datasets
    X = np.vstack(Xs)
    y = np.hstack(ys)

    features = [f"Feature{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    df["Label"] = y
    df["Domain"] = sum([[i] * num_samples for i in range(num_domains)], [])

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_rotated_moons_drain(name="Drain Rotated Two Moons"):
    """
    Returns the 2-Moons dataset from the DRAIN paper as a DistributionShiftDataset
    """
    real_path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(real_path, "data", "2-Moons", "X.npy"))
    y = np.load(os.path.join(real_path, "data", "2-Moons", "Y.npy"))

    df = pd.DataFrame(data=X, columns=["Feature1", "Feature2"])
    df["Label"] = y
    df["Domain"] = np.repeat(np.arange(10), 220)

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )


def get_shifting_sin_data(
    num_domains=10,
    num_samples=200,
    random_state=1,
    step_size=0.5,
    min_distance=0.5,
    name="Sin Classification",
):
    # Parameters
    x_range = (-10, 10)
    y_range = (-7, 7)

    # Parameters for generating data
    num_attempts_per_domain = num_samples * 10
    x_values = np.linspace(*x_range, num=num_attempts_per_domain)

    # Initialize lists to hold data
    feature1 = []
    feature2 = []
    label = []
    domain = []

    np.random.seed(random_state)

    f = lambda x: 4 * np.sin(0.5 * x + step_size * i)

    # For each domain
    for i in range(num_domains):
        # Generate shifted sine curve for this domain
        y_sine = f(x_values)

        # For each attempt in domain
        attempts = 0
        while (
            len(feature1) < (i + 1) * num_samples and attempts < num_attempts_per_domain
        ):
            attempts += 1

            # Uniformly sample x-coordinate (Feature1)
            x = np.random.uniform(*x_range)

            # Uniformly sample y-coordinate (Feature2) with a margin around sine function
            y = np.random.uniform(*y_range)

            # Calculate Euclidean distances to the sine wave
            distances = np.sqrt((x_values - x) ** 2 + (y - y_sine) ** 2)

            # If all distances are greater than margin, save data
            if np.all(distances > min_distance):
                feature1.append(x)
                feature2.append(y)

                # Label as 0 if below sine function, 1 if above
                label.append(1 if y > f(x) else 0)

                # Set domain
                domain.append(i)

    # Create DataFrame
    df = pd.DataFrame(
        {"Feature1": feature1, "Feature2": feature2, "Label": label, "Domain": domain}
    )

    return dataframe_to_distribution_shift_ds(
        name=name,
        df=df,
        target="Label",
        domain_name="Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="synthetic",
        shuffled=False,
    )
