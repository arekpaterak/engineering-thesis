from copy import deepcopy
from itertools import product

from matplotlib import pyplot as plt
import seaborn as sns

def filter_records(df, instances=None, methods=None, steps=None):
    df = deepcopy(df)
    if instances:
        df = df.loc[df['instance'].isin(instances)]
    if methods:
        df = df.loc[df['method'].isin(methods)]
    if steps:
        df = df.loc[df['steps'].isin(steps)]
    return df

def check(df, instances, methods, seeds=range(10)) -> bool:
    all_present = True
    for instance, method, seed in product(instances, methods, seeds):
        if not ((df['instance'] == instance) & (df['method'] == method) & (df['seed'] == seed)).any():
            print(f"Missing results for instance '{instance}', method '{method}', seed '{seed}'")
            all_present = False
    return all_present

def avg_over_seeds(df):
    df = deepcopy(df)
    return (
        df.groupby(['instance', 'subset', 'method', 'steps'])
        .mean(numeric_only=True)
        .reset_index()
    ).drop(columns=['seed'])


def plot_value_over_steps(
    df,
    y="objective_value",
    ylabel="Objective value",
    ylim=None,
    errorbar=('ci', 95),
    methods=None
):
    plt.figure(figsize=(6.4, 4))

    # Plot using seaborn
    sns.lineplot(
        data=df,
        x='steps',
        y=y,
        hue='method',
        marker='o',
        errorbar=errorbar,
        hue_order=methods,
    )

    if ylim:
        plt.ylim(*ylim)

    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title='', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def improvement(df):
    df = deepcopy(df)
    df['improvement'] = (
        (df['initial_objective_value'] - df['objective_value']) * 100 / df['initial_objective_value']
    ).fillna(0)
    return df


def gap(df, best_df):
    df = deepcopy(df)
    best_df = deepcopy(best_df)

    best_df = best_df.set_index('instance')['objective_value']
    df['gap'] = df.apply(
        lambda row: 100 * (row['objective_value'] - best_df[row['instance']]) / best_df[row['instance']], axis=1
    )
    return df

def plot_gap(df, best_df, steps=50, methods=None):
    sns.boxplot(
        gap(df.loc[lambda x: x.steps == steps], best_df),
        x="gap",
        y="method",
        hue="method",
        hue_order=methods,
        order=methods
    )
    plt.xlim(0)

    plt.xlabel("Gap [%]", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.tight_layout()
    plt.show()
