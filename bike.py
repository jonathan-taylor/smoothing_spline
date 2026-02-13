"""
This script performs data analysis and visualization on the bike-sharing dataset.

The script is a translation of the original R script `bike.R` to Python. 
It loads the bike-sharing data, preprocesses it, and then generates several plots 
to visualize the data and the results of linear and Poisson regression models.

The generated plots are saved in the 'plots' directory.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from glmnet import GLM
from ISLP.models import ModelSpec, contrast
from smoothing_spline import SplineFitter


def plot_smoothing_spline(dat, output_dir):
    """
    Generates and saves scatter plots with smoothing splines.
    
    This function creates two scatter plots:
    1. Hour vs. Number of Bikers
    2. Hour vs. Log(Number of Bikers)
    
    A smoothing spline is fitted to the data and plotted on top of the scatter plot.
    The data is jittered to avoid overplotting.
    
    Args:
        dat (pd.DataFrame): The input dataframe.
        output_dir (str): The directory to save the plots.
    """

    hr_numeric = pd.to_numeric(dat['hr'])
    
    # Jitter the data for better visualization
    hr_jitter = hr_numeric + np.random.uniform(-0.25, 0.25, len(hr_numeric))
    bikers_jitter = dat['bikers'] + np.random.uniform(-0.25, 0.25, len(dat['bikers']))
    log_bikers_jitter = np.log(dat['bikers']) + np.random.uniform(-0.25, 0.25, len(dat['bikers']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Hour vs. Number of Bikers
    ax1.scatter(hr_jitter, bikers_jitter, s=1, c='lightgray')
    spl = SplineFitter(df=5).fit(hr_numeric, dat['bikers'])
    ax1.plot(np.linspace(0, 23, 100), spl.predict(np.linspace(0, 23, 100)), 'b-', lw=3)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Number of Bikers")
    ax1.set_xlim(1, 24)

    # Plot 2: Hour vs. Log(Number of Bikers)
    ax2.scatter(hr_jitter, log_bikers_jitter, s=1, c='lightgray')
    spl_log = SplineFitter(df=5).fit(hr_numeric, np.log(dat['bikers']))
    ax2.plot(np.linspace(0, 23, 100), spl_log.predict(np.linspace(0, 23, 100)), 'b-', lw=3)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Log(Number of Bikers)")
    ax2.set_xlim(1, 24)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Bike-smoothingspline.pdf"))
    plt.close()


def plot_linear_model_coefficients(dat, output_dir):
    """
    Fits a linear model and plots the coefficients for month and hour.
    
    Args:
        dat (pd.DataFrame): The input dataframe.
        output_dir (str): The directory to save the plots.
    """
    mnth = contrast('mnth', method='sum')
    hr = contrast('hr', method='sum')
    weather = contrast('weathersit', method='sum')
    design = ModelSpec(['workingday', 'temp', mnth, hr, weather], intercept=False)
    mod_lm_cat = GLM(summarize=True).fit(design.transform(dat), dat['bikers'])
    S = mod_lm_cat.summary_
    mnths = [c for c in S.index if 'mnth' in c]
    hrs = [c for c in S.index if 'hr' in c]

    # Extract and plot month coefficients
    coef_months = S.loc[mnths, 'coef']
    coef_months = np.append(coef_months, -np.sum(coef_months))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(1, 13), coef_months, 'bo-')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Coefficient")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    # Extract and plot hour coefficients
    coef_hours = S.loc[hrs, 'coef']
    coef_hours = np.append(coef_hours, -np.sum(coef_hours))
    
    ax2.plot(range(24), coef_hours, 'bo-')
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Coefficient")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Bikeshare.pdf"))
    plt.close()


def analyze_weather_conditions(dat):
    """
    Analyzes and prints statistics for good and bad weather conditions.
    
    This function defines "bad weather" as hours 1, 2, or 3 in December, January, or February
    when the weather situation is classified as 3. "Good weather" is defined as hours 7, 8, or 9
    in April, May, or June when the weather situation is 1.
    
    The function then prints the mean, standard deviation, and variance of the number of bikers
    for both good and bad weather conditions.
    
    Args:
        dat (pd.DataFrame): The input dataframe.
    """
    badweather = dat[((dat['hr'].isin([1, 2, 3])) &
                      (dat['mnth'].isin([12, 1, 2])) &
                      (dat['weathersit'] == 3))]
    
    print("Bad Weather Statistics:")
    print(f"Mean bikers: {badweather['bikers'].mean()}")
    print(f"Std dev bikers: {badweather['bikers'].std()}")
    print(f"Variance bikers: {badweather['bikers'].var()}")

    goodweather = dat[((dat['hr'].isin([7, 8, 9])) &
                       (dat['mnth'].isin([4, 5, 6])) &
                       (dat['weathersit'] == 1))]

    print("\nGood Weather Statistics:")
    print(f"Mean bikers: {goodweather['bikers'].mean()}")
    print(f"Std dev bikers: {goodweather['bikers'].std()}")
    print(f"Variance bikers: {goodweather['bikers'].var()}")


def plot_poisson_model_coefficients(dat, output_dir):
    """
    Fits a Poisson model and plots the coefficients for month and hour.
    
    Args:
        dat (pd.DataFrame): The input dataframe.
        output_dir (str): The directory to save the plots.
    """
    mnth = contrast('mnth', method='sum')
    hr = contrast('hr', method='sum')
    weather = contrast('weathersit', method='sum')
    design = ModelSpec(['workingday', 'temp', mnth, hr, weather], intercept=False)
    X = design.fit_transform(dat)
    mod_lm_cat_pois = GLM(family=sm.families.Poisson(), summarize=True).fit(X, dat['bikers'])

    S = mod_lm_cat_pois.summary_
    mnths = [c for c in S.index if 'mnth' in c]
    hrs = [c for c in S.index if 'hr' in c]

    # Extract and plot month coefficients
    coef_months = S.loc[mnths, 'coef']
    coef_months = np.append(coef_months, -np.sum(coef_months))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(1, 13), coef_months, 'bo-')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Coefficient")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    # Extract and plot hour coefficients
    coef_hours = S.loc[hrs, 'coef']
    coef_hours = np.append(coef_hours, -np.sum(coef_hours))
    
    ax2.plot(range(24), coef_hours, 'bo-')
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Coefficient")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Bikeshare-2.pdf"))
    plt.close()


def main():
    """
    Main function to run the analysis.
    
    This function loads the bike-sharing dataset, performs some preprocessing,
    and then calls the other functions to generate plots and analyze the data.
    """
    # Load the data
    dat = pd.read_csv("../bike.csv")
    dat['bikers'] = dat['cnt']

    # Convert columns to categorical
    dat['hr'] = pd.Categorical(dat['hr'])
    dat['mnth'] = pd.Categorical(dat['mnth'])
    
    # Create the output directory if it doesn't exist
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Visualization 1: Smoothing Spline ---
    plot_smoothing_spline(dat, output_dir)

    # --- Linear Regression Model ---
    plot_linear_model_coefficients(dat, output_dir)
    
    # --- Data Subsetting and Analysis ---
    analyze_weather_conditions(dat)

    # --- Poisson Regression Model ---
    plot_poisson_model_coefficients(dat, output_dir)


if __name__ == '__main__':
    main()
